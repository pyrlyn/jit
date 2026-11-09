//! x86-64 Compiler: IR (`core/ir.zig`) -> register allocation (`core/regalloc.zig`, one run per
//! register class, `core/reg_classes.zig`) -> Builder nodes (`builder.zig`) -> machine code.
//! Like AsmJit's x86::Compiler but over a small typed IR instead of virtual-register operands.
//!
//! Conventions (host C ABI: SysV on macOS/Linux, Win64 on Windows):
//! - GP registers: all but rsp, rbp (frame) and r10/r11 (scratch) are allocatable. The
//!   allocator numbers them in `gp_order`, caller-saved first, so its lowest-free choice keeps
//!   call-free code in caller-saved registers (leaf functions push nothing) while values live
//!   across a call take callee-saved ones or are saved around it (`Options.split_at_calls`).
//!   Fixed-register needs are allocator constraints, not reserved registers: `idiv` clobbers
//!   rax:rdx and a variable shift rcx (`regalloc.Config.clobbers`); hints
//!   (`regalloc.Config.hints`) steer parameters, call arguments and results, `ret`, the dividend
//!   and the shift count into their fixed registers and a two-address result into its left
//!   operand's register. Parameter and argument moves are parallel moves (cycles via r10).
//! - Lowering is dst-direct: `dst = a op b` is `op dst, b` when dst == a, `mov dst, a; op dst,
//!   b` otherwise, `op dst, a` for commutative ops with dst == b (`neg dst; add dst, a` for
//!   sub), `lea` for adds into a third register. r10 only for spilled destinations (written
//!   through to the slot) and the remaining conflicts; spilled operands are memory operands.
//! - Constants: a vreg is known at a use when it was defined by a constant-valued op (`const`,
//!   a copy of a known value, `arith` of two known values that cannot trap) earlier in the same
//!   block, or when its only definition is a `const` in block 0 that precedes every block-0 use
//!   (block 0 runs before any other block). Known operands become immediates (`op r, imm32`,
//!   shifts by `count & 63`, `lea`, `imul r, r, imm32`, `shl` for a power-of-two `mul`, `cmp r,
//!   imm32` or `test r, r` for 0, `mov qword [m], imm32`) or are materialized directly where
//!   they are needed (call arguments, `ret`, the dividend, ...). A vreg whose definitions are
//!   all constant-valued and whose uses all fold gets no register and no code
//!   (`Stats.excluded`). Zero is materialized as `xor r32, r32`, never between a cmp and its
//!   jcc. Divisions by zero or minInt / -1 are never folded (they trap at run time).
//! - Branches follow the block layout (IR order): no jump to the next block; a `cbr` whose
//!   `then` is the next block becomes the inverted jcc to `else`. Branches are rel8 when in range.
//! - Frame: `push rbp; mov rbp, rsp` only with stack slots or stack parameters; callee-saved
//!   registers are pushed only when used; a leaf function without either has no frame at all.
//!   The epilog is inlined at every `ret`. rsp is 16-aligned at calls.
//! - FP class: scalar f32/f64 and 128-bit vectors (v4i32/v2i64/v4f32/v2f64) in xmm registers,
//!   two scratch registers (`fs0`/`fs1`). With vectors, FP slots are 16 bytes at 16-aligned rbp
//!   offsets (aligned movaps spills and m128 operands). Float values crossing a call are
//!   spilled. Vector code is SSE2 except `pmulld` (SSE4.1), used only when the build gate
//!   (`-Dx86-features`) allows it and the host CPU has it (`Options.sse41`); v4i32 `mul`
//!   otherwise uses the SSE2 pmuludq/pshufd sequence and v2i64 `mul` scalar `imul` per lane.
const std = @import("std");
const x = @import("assembler.zig");
const bld = @import("builder.zig");
const ir = @import("../core/ir.zig");
const ra = @import("../core/regalloc.zig");
const rc = @import("../core/reg_classes.zig");
const cpu = @import("cpu.zig");
const CodeBuffer = @import("../core/code_buffer.zig").CodeBuffer;
const Gp = x.Gp;
const Xmm = bld.Xmm;

const cc = x.CallConv.host;
const win = cc.shadow != 0;
const s0: Gp = .r10;
const s1: Gp = .r11;

/// Allocator numbering of the GP registers (allocator register i is `gp_order[i]`): the
/// caller-saved ones first (rax, rdx, rcx last among them: division and shifts need them),
/// then the callee-saved ones.
const gp_order: [12]Gp = if (win)
    .{ .r8, .r9, .rdx, .rcx, .rax, .rbx, .rsi, .rdi, .r12, .r13, .r14, .r15 }
else
    .{ .r8, .r9, .rdi, .rsi, .rdx, .rcx, .rax, .rbx, .r12, .r13, .r14, .r15 };
const n_caller_saved = if (win) 5 else 7;
/// Callee-saved registers in allocator numbering.
const abs_callee_saved: ra.RegMask = ((1 << gp_order.len) - 1) & ~@as(ra.RegMask, (1 << n_caller_saved) - 1);

fn bit(r: Gp) ra.RegMask {
    return @as(ra.RegMask, 1) << @backingInt(r);
}

/// Allocatable GP registers (physical register bits).
pub const allocatable: ra.RegMask = blk: {
    var m: ra.RegMask = 0;
    for (gp_order) |r| m |= bit(r);
    break :blk m;
};
/// Allocatable callee-saved GP registers (physical register bits).
pub const callee_saved: ra.RegMask = blk: {
    var m: ra.RegMask = 0;
    for (gp_order[n_caller_saved..]) |r| m |= bit(r);
    break :blk m;
};
const abs_index: [16]u8 = blk: {
    var a: [16]u8 = @splat(ra.no_reg);
    for (gp_order, 0..) |r, i| a[@backingInt(r)] = i;
    break :blk a;
};
fn absOf(r: Gp) u8 {
    return abs_index[@backingInt(r)];
}
fn gpOf(i: u8) Gp {
    return gp_order[i];
}
/// Physical register bits -> allocator numbering (registers outside `gp_order` are dropped).
fn toAbs(m: ra.RegMask) ra.RegMask {
    var out: ra.RegMask = 0;
    for (gp_order, 0..) |r, i| if (m & bit(r) != 0) {
        out |= @as(ra.RegMask, 1) << @intCast(i);
    };
    return out;
}

/// FP class (scalar floats and 128-bit vectors in xmm registers, `core/reg_classes.zig`).
/// Every xmm register is caller-saved on SysV; on Win64 xmm6..15 are callee-saved, so only
/// xmm0..3 are allocated there (xmm4/5 are scratch).
pub const fp_allocatable: ra.RegMask = if (win) 0b1111 else 0x3FFF; // xmm0..3 / xmm0..13
const fs0: Xmm = if (win) 4 else 14;
const fs1: Xmm = if (win) 5 else 15;

pub const Options = struct {
    /// Restrict the allocatable GP set (physical register bits; rsp, rbp, r10 and r11 are
    /// never allocated). Spill stress tests.
    allocatable: ra.RegMask = allocatable,
    /// Keep call-crossing values in caller-saved registers when no callee-saved one is free,
    /// saving/restoring them around each call (`regalloc.Config.split_at_calls`).
    split_at_calls: bool = true,
    shorten_branches: bool = true,
    algo: ra.Algo = .linear,
    /// Per-block register segments for spilled values (`regalloc.Config.split`).
    split: bool = true,
    /// Restrict the FP (xmm) allocatable set (spill stress tests; scratch registers ignored).
    fp_allocatable: ra.RegMask = fp_allocatable,
    /// v4i32 `mul` with SSE4.1 `pmulld`. null: when the build gate allows SSE4.1 and the host
    /// CPU has it (`cpu.host()`); false forces the SSE2 sequence.
    sse41: ?bool = null,
};

pub const Stats = struct {
    spills: u32 = 0,
    slots: u32 = 0,
    nodes: usize = 0,
    /// Operands folded into an immediate (`add r, imm`, `cmp r, imm`, `imul r, r, imm`, ...).
    folded_imm: u32 = 0,
    /// Spilled operands used directly as memory (`add r, [rbp-off]`, `addsd x, [rbp-off]`).
    folded_mem: u32 = 0,
    /// Register segments given to spilled values (live-range splitting).
    segments: u32 = 0,
    /// Copies removed by coalescing (`.graph`).
    coalesced: u32 = 0,
    /// Constant vregs kept out of register allocation (every use folded).
    excluded: u32 = 0,
};

/// Compiles `f` into `b` (a Builder). Run passes and serialize separately, or use `compile`.
/// Functions with block parameters are first rewritten by `ir.Func.lowerBlockParams`.
pub fn lower(gpa: std.mem.Allocator, f0: *const ir.Func, b: *bld.Builder, opts: Options) !Stats {
    return lowerInto(gpa, f0, b, opts);
}

fn fitsImm(k: i64) bool {
    return std.math.cast(i32, k) != null;
}
/// `log2(k)` for a positive power of two.
fn pow2Shift(k: i64) ?u6 {
    if (k <= 0 or k & (k - 1) != 0) return null;
    return @intCast(@ctz(k));
}
fn divTraps(a: i64, b: i64) bool {
    return b == 0 or (a == std.math.minInt(i64) and b == -1);
}

/// Known constant values at the current program point (module doc): block-local values by
/// generation stamp (no per-block reset) and function-wide constants (`glob`).
const Known = struct {
    glob: []?i64,
    val: []i64,
    gen: []u32,
    cur: u32 = 0,

    fn init(A: std.mem.Allocator, n: usize) !Known {
        const k: Known = .{ .glob = try A.alloc(?i64, n), .val = try A.alloc(i64, n), .gen = try A.alloc(u32, n) };
        @memset(k.glob, null);
        @memset(k.gen, 0);
        return k;
    }
    fn reset(k: *Known) void {
        @memset(k.gen, 0);
        k.cur = 0;
    }
    fn block(k: *Known) void {
        k.cur += 1;
    }
    fn get(k: *const Known, v: ir.VReg) ?i64 {
        if (k.gen[v] == k.cur) return k.val[v];
        return k.glob[v];
    }
    fn set(k: *Known, v: ir.VReg, c: ?i64) void {
        if (c) |val| {
            k.gen[v] = k.cur;
            k.val[v] = val;
        } else k.gen[v] = 0;
    }
};

/// Immediate operands of an `arith` (shared by the analysis and the lowering, so a constant
/// left out of register allocation is never read from a register).
const ArithPlan = struct { a: ?i64 = null, b: ?i64 = null, value: ?i64 = null };

fn foldsArith(op: ir.Arith, k: i64, left: bool) bool {
    return switch (op) {
        .add, .@"and", .@"or", .xor => fitsImm(k), // imm32 (commutative: also the left operand)
        .sub => left or fitsImm(k), // left: materialized in the result register
        .mul => fitsImm(k) or pow2Shift(k) != null,
        .shl, .shr, .sar, .sdiv, .srem => true, // imm8 count; materialized dividend/divisor
    };
}

fn planArith(c: anytype, k: *const Known) ArithPlan {
    const ka = k.get(c.a);
    const kb = k.get(c.b);
    if (ka != null and kb != null) {
        // Undefined divisions keep the instruction (it traps), with both operands materialized.
        if ((c.op == .sdiv or c.op == .srem) and divTraps(ka.?, kb.?)) return .{ .a = ka, .b = kb };
        return .{ .value = ir.arith(c.op, ka.?, kb.?) };
    }
    if (kb) |v| if (foldsArith(c.op, v, false)) return .{ .b = v };
    if (ka) |v| if (foldsArith(c.op, v, true)) return .{ .a = v };
    return .{};
}

/// `cbr`: a static branch when both operands are known, else at most one imm32 (`b`, or `a`
/// with swapped operands).
const CmpPlan = struct { a: ?i64 = null, b: ?i64 = null, static: ?bool = null };

fn planCmp(c: anytype, k: *const Known) CmpPlan {
    const ka = k.get(c.a);
    const kb = k.get(c.b);
    if (ka != null and kb != null) return .{ .static = ir.cond(c.cond, ka.?, kb.?) };
    if (kb) |v| if (fitsImm(v)) return .{ .b = v };
    if (ka) |v| if (fitsImm(v)) return .{ .a = v };
    return .{};
}

/// Value of the vreg defined by `o` when it is a compile-time constant.
fn constDef(o: *const ir.Op, k: *const Known) ?i64 {
    return switch (o.*) {
        .@"const" => |c| c.imm,
        .copy => |c| k.get(c.src), // float vregs are never known
        .arith => |c| planArith(c, k).value,
        else => null,
    };
}

/// Register-allocation input derived from the IR in one pass (O(ops)).
const Analysis = struct {
    known: Known,
    /// Vreg types for the GP run: excluded constants are marked as FP so that neither run
    /// allocates them.
    gtypes: []ir.Type,
    excluded: u32,
    hints: []ra.Hint,
    fhints: []ra.Hint,
    clobbers: []ra.Clobber,
    has_vectors: bool,
    /// Positions of the calls, ascending.
    calls: []const u32,
};

fn analyze(A: std.mem.Allocator, f: *const ir.Func, types: []const ir.Type) !Analysis {
    const n = types.len;
    var k = try Known.init(A, n);
    // Function-wide constants: one definition, a `const` in block 0 before any block-0 use.
    {
        const ndefs = try A.alloc(u8, n);
        @memset(ndefs, 0);
        for (0..f.params) |v| ndefs[v] = 1;
        const early = try A.alloc(bool, n);
        @memset(early, false);
        var candidates = false;
        if (f.blocks.items.len != 0) for (f.blocks.items[0].ops.items) |o| {
            // A use before the vreg's `const` in block 0 reads another value.
            var buf: [ir.max_args]ir.VReg = undefined;
            for (o.uses(&buf)) |u| if (k.glob[u] == null) {
                early[u] = true;
            };
            if (o.def()) |d| {
                ndefs[d] +|= 1;
                if (o == .@"const" and ndefs[d] == 1 and !early[d]) {
                    k.glob[d] = o.@"const".imm;
                    candidates = true;
                }
            }
        };
        // Definitions in the other blocks only matter for the candidates.
        if (candidates) for (f.blocks.items[1..]) |blk| for (blk.ops.items) |o| {
            if (o.def()) |d| ndefs[d] +|= 1;
        };
        for (k.glob, 0..) |*g, v| if (ndefs[v] != 1 or early[v]) {
            g.* = null;
        };
    }

    const needs = try A.alloc(bool, n); // a GP use needs a register
    @memset(needs, false);
    const varying = try A.alloc(bool, n); // a definition is not constant-valued
    @memset(varying, false);
    for (0..f.params) |v| varying[v] = true;
    const hints = try A.alloc(ra.Hint, n);
    @memset(hints, .{});
    const fhints = try A.alloc(ra.Hint, n);
    @memset(fhints, .{});
    var clobbers: std.ArrayList(ra.Clobber) = .empty;
    var calls: std.ArrayList(u32) = .empty;
    var has_vectors = false;
    for (types) |tp| has_vectors = has_vectors or tp.isVector();

    const H = struct {
        fn reg(h: []ra.Hint, v: ir.VReg, r: Gp) void {
            if (h[v].reg == ra.no_reg) h[v].reg = absOf(r);
        }
        fn vreg(h: []ra.Hint, v: ir.VReg, w: ir.VReg) void {
            if (h[v].vreg == std.math.maxInt(ra.VReg)) h[v].vreg = w;
        }
    };
    for (0..@min(f.params, cc.args.len)) |i| H.reg(hints, @intCast(i), cc.args[i]);
    const rax_rdx = (@as(ra.RegMask, 1) << @intCast(absOf(.rax))) | (@as(ra.RegMask, 1) << @intCast(absOf(.rdx)));
    const rcx_m = @as(ra.RegMask, 1) << @intCast(absOf(.rcx));

    var pos: u32 = 0;
    for (f.blocks.items) |blk| {
        k.block();
        for (blk.ops.items) |*o| {
            pos += 1;
            const ap: ArithPlan = if (o.* == .arith) planArith(o.arith, &k) else .{};
            const cd = if (o.* == .arith) ap.value else constDef(o, &k);
            switch (o.*) {
                .@"const", .fconst, .br, .bra, .cbra => {},
                .copy => |c| if (types[c.dst] == .i64) {
                    if (cd == null) {
                        needs[c.src] = true;
                        H.vreg(hints, c.dst, c.src);
                    }
                } else H.vreg(fhints, c.dst, c.src),
                .arith => |c| if (cd == null) {
                    const p = ap;
                    if (p.a == null) needs[c.a] = true;
                    if (p.b == null) needs[c.b] = true;
                    switch (c.op) {
                        .sdiv, .srem => {
                            try clobbers.append(A, .{ .pos = pos, .mask = rax_rdx });
                            H.reg(hints, c.a, .rax);
                            H.reg(hints, c.dst, if (c.op == .sdiv) .rax else .rdx);
                        },
                        .shl, .shr, .sar => if (p.b == null) {
                            try clobbers.append(A, .{ .pos = pos, .mask = rcx_m });
                            H.reg(hints, c.b, .rcx);
                        },
                        else => {},
                    }
                    const comm = switch (c.op) {
                        .add, .mul, .@"and", .@"or", .xor => true,
                        else => false,
                    };
                    if (p.a == null) H.vreg(hints, c.dst, c.a) else if (comm and p.b == null) H.vreg(hints, c.dst, c.b);
                },
                .load => |c| if (k.get(c.base) == null) {
                    needs[c.base] = true;
                },
                .store => |c| {
                    if (k.get(c.base) == null) needs[c.base] = true;
                    if (c.ty == .i64) if (k.get(c.src)) |v| {
                        if (!fitsImm(v)) needs[c.src] = true;
                    } else {
                        needs[c.src] = true;
                    };
                },
                .typed => |c| H.vreg(fhints, c.dst, c.a),
                .fcmp, .extract => {},
                .cvt => |c| if (c.from == .i64) {
                    if (k.get(c.src) == null) needs[c.src] = true;
                } else if (c.to != .i64) H.vreg(fhints, c.dst, c.src),
                .splat => |c| if (c.ty.lane() == .i64) {
                    if (k.get(c.src) == null) needs[c.src] = true;
                } else H.vreg(fhints, c.dst, c.src),
                .call => |c| {
                    try calls.append(A, pos);
                    for (c.args, 0..) |a, i| if (k.get(a) == null) {
                        needs[a] = true;
                        if (i < cc.args.len) H.reg(hints, a, cc.args[i]);
                    };
                    H.reg(hints, c.dst, .rax);
                },
                .cbr => |c| {
                    const p = planCmp(c, &k);
                    if (p.static == null) {
                        if (p.a == null) needs[c.a] = true;
                        if (p.b == null) needs[c.b] = true;
                    }
                },
                .ret => |v| if (k.get(v) == null) {
                    needs[v] = true;
                    H.reg(hints, v, .rax);
                },
            }
            if (o.def()) |d| {
                if (cd == null) varying[d] = true;
                k.set(d, cd);
            }
        }
    }
    const gtypes = try A.dupe(ir.Type, types);
    var excluded: u32 = 0;
    for (gtypes, 0..) |*tp, v| if (tp.* == .i64 and !needs[v] and !varying[v]) {
        tp.* = .f64;
        excluded += 1;
    };
    k.reset();
    return .{ .known = k, .gtypes = gtypes, .excluded = excluded, .hints = hints, .fhints = fhints, .clobbers = clobbers.items, .has_vectors = has_vectors, .calls = calls.items };
}

fn lowerInto(backing: std.mem.Allocator, f0: *const ir.Func, b: anytype, opts: Options) !Stats {
    var scratch_bytes: [2048]u8 = undefined;
    var temp_alloc = @import("../core/scratch.zig").Scratch.init(&scratch_bytes, backing);
    const gpa = temp_alloc.allocator();
    try f0.verify();
    var lowered: ?ir.Func = if (f0.hasBlockParams()) try f0.lowerBlockParams(gpa) else null;
    defer if (lowered) |*g| g.deinit();
    const f: *const ir.Func = if (lowered) |*g| g else f0;
    if (f.params > ir.max_args) return error.TooManyParams;

    // ---- analysis and register allocation (one run per register class) ----
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const A = arena.allocator();
    const types = try rc.checkWith(A, f, .{ .vectors = true });
    const an = try analyze(A, f, types);
    const gmask = toAbs(opts.allocatable);
    const rcfg: ra.Config = .{
        .allocatable = gmask,
        .callee_saved = abs_callee_saved & gmask,
        .split_at_calls = opts.split_at_calls,
        .algo = opts.algo,
        .split = opts.split,
        .hints = an.hints,
        .clobbers = an.clobbers,
    };
    var res = try rc.allocateClass(gpa, f, an.gtypes, .gp, true, rcfg);
    defer res.deinit(gpa);
    const fmask = opts.fp_allocatable & ~((@as(ra.RegMask, 1) << fs0) | (@as(ra.RegMask, 1) << fs1)) & 0xFFFF;
    var fres = try rc.allocateClass(gpa, f, types, .fp, false, .{ .allocatable = fmask, .hints = an.fhints });
    defer fres.deinit(gpa);
    // Call positions, and per call the vregs saved/restored around it (split ranges, ascending;
    // `call_saves[call_start[i]..call_start[i + 1]]`). Each save vreg visits only the calls its
    // interval crosses (a contiguous range of `call_pos`).
    const call_pos = an.calls;
    const call_start = try A.alloc(u32, call_pos.len + 1);
    @memset(call_start, 0);
    var nsaves: u32 = 0;
    var call_saves_buf: []ir.VReg = &.{};
    for (0..2) |pass| {
        for (res.save_slot, 0..) |sl, vi| if (sl != null) {
            const v: ir.VReg = @intCast(vi);
            const iv = res.intervals[v];
            var i = std.sort.partitionPoint(u32, call_pos, iv.start, struct {
                fn before(start: u32, c: u32) bool {
                    // Same rule as regalloc.crossesCall: a call at the interval's start
                    // (a value live into a block that begins with the call) is crossed.
                    return 2 * c < start;
                }
            }.before);
            while (i < call_pos.len and res.savedAcross(v, call_pos[i])) : (i += 1) {
                if (pass == 0) call_start[i + 1] += 1 else {
                    call_saves_buf[call_start[i]] = v;
                    call_start[i] += 1;
                }
            }
        };
        if (pass == 0) {
            for (call_start[1..], 0..) |*c, i| c.* += call_start[i];
            nsaves = call_start[call_pos.len];
            call_saves_buf = try A.alloc(ir.VReg, nsaves);
        }
    }
    // The fill pass advanced every start to the next call's start: shift back.
    std.mem.copyBackwards(u32, call_start[1..], call_start[0 .. call_start.len - 1]);
    call_start[0] = 0;

    // ---- frame ----
    const used = res.usedRegs();
    var saved: std.ArrayList(Gp) = .empty;
    for (gp_order[n_caller_saved..], n_caller_saved..) |r, i| if (used & (@as(ra.RegMask, 1) << @intCast(i)) != 0) try saved.append(A, r);
    const nsaved: u32 = @intCast(saved.items.len);
    const has_calls = call_pos.len != 0;
    const fsz: u32 = if (an.has_vectors) 16 else 8;
    const use_rbp = res.num_slots + fres.num_slots != 0 or f.params > cc.args.len;
    // GP slot k at [rbp - 8*(nsaved+1+k)]; FP slot k at [rbp - fbase - fsz*(k+1)] (16-aligned
    // for vectors: rbp is 16-aligned after `push rbp`).
    const fbase = std.mem.alignForward(u32, 8 * (nsaved + res.num_slots), fsz);
    const adj: u32 = if (use_rbp)
        std.mem.alignForward(u32, fbase + fsz * fres.num_slots, 16) - 8 * nsaved
    else if (has_calls and nsaved % 2 == 0) 8 else 0;
    const gate_sse41 = comptime x.build_gate == null or x.build_gate.?.contains(.SSE4_1);
    const sse41 = gate_sse41 and (opts.sse41 orelse if (cpu.host()) |h| h.contains(.SSE4_1) else false);

    const seg_next = try A.alloc(u32, if (res.seg_start.len == 0) 0 else res.seg_start.len - 1);
    if (seg_next.len != 0) @memcpy(seg_next, res.seg_start[0..seg_next.len]);
    const labels = try A.alloc(bld.Label, f.blocks.items.len);
    for (labels) |*l| l.* = try b.newLabel();
    var s: Lowering(@TypeOf(b)) = .{
        .b = b,
        .types = types,
        .res = &res,
        .fres = &fres,
        .k = an.known,
        .nsaved = nsaved,
        .fbase = fbase,
        .fsz = fsz,
        .use_rbp = use_rbp,
        .adj = adj,
        .saved = saved.items,
        .call_saves = call_saves_buf,
        .call_start = call_start,
        .labels = labels,
        .sse41 = sse41,
        .seg_next = seg_next,
    };

    // ---- prolog ----
    try b.comment(f.name);
    if (use_rbp) {
        try b.emit(.push, .{.rbp});
        try b.emit(.movRR, .{ .rbp, .rsp });
    }
    for (saved.items) |r| try b.emit(.push, .{r});
    if (adj != 0) try b.emit(.subRI, .{ .rsp, @intCast(adj) });
    try s.params(f.params);
    // Values read before any def on some path are 0 (IR semantics, as in `ir.interpret`).
    for (res.entry_undef) |v| try s.setConst(v, 0);
    var fzero = false;
    for (fres.entry_undef) |v| switch (fres.locs[v]) {
        .reg => |r| try b.emit(.sseRR, .{ .xorps, @intCast(r), @intCast(r) }),
        .stack => |sl| {
            if (!fzero) try b.emit(.sseRR, .{ .xorps, fs0, fs0 });
            fzero = true;
            try b.emit(.sseMR, .{ movFor(types[v], true), s.fslot(sl), fs0 });
        },
        .none => {},
    };

    for (f.blocks.items, 0..) |blk, bi| {
        try b.bind(labels[bi]);
        s.k.block();
        s.next = @intCast(bi + 1);
        for (blk.ops.items) |*o| {
            s.pos += 1;
            // `ra.Result.loadsAt` without its binary search: positions only grow.
            while (s.load_next < res.load_order.len) {
                const sg = res.segs[res.load_order[s.load_next]];
                if (sg.first > s.pos) break;
                s.load_next += 1;
                if (sg.first == s.pos) try s.loadSeg(sg.vreg, sg.reg);
            }
            const ap: ArithPlan = if (o.* == .arith) planArith(o.arith, &s.k) else .{};
            const cd = if (o.* == .arith) ap.value else constDef(o, &s.k);
            try s.lowerOp(o, cd, ap);
            if (o.def()) |d| s.k.set(d, cd);
        }
    }
    return .{ .spills = res.spills, .slots = res.num_slots, .nodes = b.count(), .folded_imm = s.folded_imm, .folded_mem = s.folded_mem, .segments = @intCast(res.segs.len), .coalesced = res.coalesced, .excluded = an.excluded };
}

/// Memory move for an FP-class value of type `tp` (slots are aligned, IR memory is not).
fn movFor(tp: ir.Type, aligned: bool) bld.SseMov {
    return switch (tp) {
        .f32 => .movss,
        .f64 => .movsd,
        .v4i32, .v2i64 => if (aligned) .movaps else .movdqu,
        else => if (aligned) .movaps else .movups,
    };
}

fn condOf(c: ir.CondCode) x.Cond {
    return switch (c) {
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
}
/// The condition with its operands swapped (`a < b` == `b > a`).
fn swapCond(c: x.Cond) x.Cond {
    return switch (c) {
        .l => .g,
        .g => .l,
        .le => .ge,
        .ge => .le,
        .b => .a,
        .a => .b,
        .be => .ae,
        .ae => .be,
        else => c,
    };
}
fn invertCond(c: x.Cond) x.Cond {
    return @fromBackingInt(@backingInt(c) ^ 1);
}

const Loc = union(enum) { reg: Gp, mem: x.Mem };
const FLoc = union(enum) { reg: Xmm, mem: x.Mem };
const Src = union(enum) { reg: Gp, imm: i64, mem: x.Mem };
const Move = struct { dst: Gp, src: Src };

fn Lowering(comptime B: type) type {
    return struct {
        const Self = @This();
        b: B,
        types: []const ir.Type,
        res: *const ra.Result,
        fres: *const ra.Result,
        k: Known,
        nsaved: u32,
        fbase: u32,
        fsz: u32,
        use_rbp: bool,
        adj: u32,
        saved: []const Gp,
        /// Vregs saved around call number i: `call_saves[call_start[i]..call_start[i + 1]]`.
        call_saves: []const ir.VReg,
        call_start: []const u32,
        /// Calls lowered so far.
        ncall: u32 = 0,
        labels: []const bld.Label,
        sse41: bool,
        /// `segReg` cursors: index of the next segment of each vreg in `res.segs`.
        seg_next: []u32,
        /// Next entry of `res.load_order` (segments loaded at their first position, by position).
        load_next: u32 = 0,
        /// Instruction position (regalloc numbering: 0 is the parameter definitions).
        pos: u32 = 0,
        /// Index of the block emitted after the current one.
        next: u32 = 0,
        folded_imm: u32 = 0,
        folded_mem: u32 = 0,

        /// Register segment of spilled `v` at the current position. Positions only grow while
        /// lowering, so a per-vreg cursor makes this amortized O(1) (`ra.Result.regAt` scans
        /// every segment of the vreg: quadratic for values spilled across many blocks).
        fn segReg(s: *Self, v: ir.VReg) ?Gp {
            const st = s.res.seg_start;
            if (st.len == 0) return null;
            var i = s.seg_next[v];
            const end = st[v + 1];
            while (i < end and s.res.segs[i].last < s.pos) i += 1;
            s.seg_next[v] = i;
            if (i < end and s.res.segs[i].first <= s.pos) return gpOf(s.res.segs[i].reg);
            return null;
        }
        fn gslot(s: *const Self, k: u32) x.Mem {
            return x.ptr(.rbp, -@as(i32, @intCast(8 * (s.nsaved + 1 + k))));
        }
        fn fslot(s: *const Self, k: u32) x.Mem {
            return x.ptr(.rbp, -@as(i32, @intCast(s.fbase + s.fsz * (k + 1))));
        }

        // ---- GP locations ----
        /// Where `v` is read at the current position (a register segment of a spilled value
        /// is a register).
        fn loc(s: *Self, v: ir.VReg) !Loc {
            return switch (s.res.locs[v]) {
                .reg => |r| .{ .reg = gpOf(r) },
                .stack => |k| if (s.segReg(v)) |r| .{ .reg = r } else .{ .mem = s.gslot(k) },
                // Excluded constants are only read through fold plans; undefined reads are
                // live into the entry block and have a location.
                .none => error.UnallocatedOperand,
            };
        }
        /// Register holding `v` (spilled values are loaded into `scratch`).
        fn src(s: *Self, v: ir.VReg, scratch: Gp) !Gp {
            return switch (try s.loc(v)) {
                .reg => |r| r,
                .mem => |m| blk: {
                    try s.b.emit(.movRM, .{ scratch, m });
                    break :blk scratch;
                },
            };
        }
        /// Register to compute the definition of `v` in: its register, its segment register or
        /// r10 (then written to the slot by `put`).
        fn dst(s: *Self, v: ir.VReg) Gp {
            return switch (s.res.locs[v]) {
                .reg => |r| gpOf(r),
                .stack => s.segReg(v) orelse s0,
                .none => s0,
            };
        }
        /// Defines `v` from register `r` (written through to the slot of a spilled value).
        fn put(s: *Self, v: ir.VReg, r: Gp) !void {
            switch (s.res.locs[v]) {
                .reg => |d| if (gpOf(d) != r) try s.b.emit(.movRR, .{ gpOf(d), r }),
                .stack => |k| {
                    if (s.segReg(v)) |d| if (d != r) try s.b.emit(.movRR, .{ d, r });
                    try s.b.emit(.movMR, .{ s.gslot(k), r });
                },
                .none => {},
            }
        }
        fn movTo(s: *Self, r: Gp, l: Loc) !void {
            switch (l) {
                .reg => |q| if (q != r) try s.b.emit(.movRR, .{ r, q }),
                .mem => |m| try s.b.emit(.movRM, .{ r, m }),
            }
        }
        /// `r = k` (`xor r32, r32` for 0: callers never materialize between a cmp and its jcc).
        fn materialize(s: *Self, r: Gp, k: i64) !void {
            if (k == 0) try s.b.emit(.zero32, .{r}) else try s.b.emit(.movRI, .{ r, k });
        }
        /// Defines `v` as the constant `k` (nothing for an excluded constant).
        fn setConst(s: *Self, v: ir.VReg, k: i64) !void {
            switch (s.res.locs[v]) {
                .reg => |r| try s.materialize(gpOf(r), k),
                .stack => |sl| if (s.segReg(v)) |r| {
                    try s.materialize(r, k);
                    try s.b.emit(.movMR, .{ s.gslot(sl), r });
                } else if (fitsImm(k)) {
                    try s.b.emit(.movMI, .{ s.gslot(sl), @intCast(k) });
                } else {
                    try s.b.emit(.movRI, .{ s0, k });
                    try s.b.emit(.movMR, .{ s.gslot(sl), s0 });
                },
                .none => {},
            }
        }
        fn loadSeg(s: *Self, v: ir.VReg, r: u8) !void {
            try s.b.emit(.movRM, .{ gpOf(r), s.gslot(s.res.locs[v].stack) });
        }
        /// `w = w op l` for add/sub/and/or/xor/mul.
        fn opRL(s: *Self, op: ir.Arith, w: Gp, l: Loc) !void {
            switch (l) {
                .reg => |r| if (op == .mul) try s.b.emit(.imulRR, .{ w, r }) else try s.b.emit(.aluRR, .{ aluOf(op), w, r }),
                .mem => |m| {
                    s.folded_mem += 1;
                    if (op == .mul) try s.b.emit(.imulRM, .{ w, m }) else try s.b.emit(.aluRM, .{ aluOf(op), w, m });
                },
            }
        }
        fn aluOf(op: ir.Arith) x.Alu {
            return switch (op) {
                .add => .add,
                .sub => .sub,
                .@"and" => .@"and",
                .@"or" => .@"or",
                .xor => .xor,
                else => unreachable,
            };
        }

        // ---- FP locations ----
        fn floc(s: *const Self, v: ir.VReg) FLoc {
            return switch (s.fres.locs[v]) {
                .reg => |r| .{ .reg = @intCast(r) },
                .stack => |k| .{ .mem = s.fslot(k) },
                // Never defined nor live into the entry block: never read.
                .none => .{ .reg = fs1 },
            };
        }
        fn fsrc(s: *Self, v: ir.VReg, scratch: Xmm) !Xmm {
            return switch (s.floc(v)) {
                .reg => |r| r,
                .mem => |m| blk: {
                    try s.b.emit(.sseRM, .{ movFor(s.types[v], true), scratch, m });
                    break :blk scratch;
                },
            };
        }
        fn fdst(s: *const Self, v: ir.VReg) Xmm {
            return switch (s.fres.locs[v]) {
                .reg => |r| @intCast(r),
                else => fs0,
            };
        }
        fn fput(s: *Self, v: ir.VReg, r: Xmm) !void {
            switch (s.fres.locs[v]) {
                .reg => |d| if (d != r) try s.b.emit(.sseRR, .{ .movaps, @intCast(d), r }),
                .stack => |k| try s.b.emit(.sseMR, .{ movFor(s.types[v], true), s.fslot(k), r }),
                .none => {},
            }
        }
        /// `r = l` for an FP value of type `tp`.
        fn fmov(s: *Self, r: Xmm, l: FLoc, tp: ir.Type) !void {
            switch (l) {
                .reg => |q| if (q != r) try s.b.emit(.sseRR, .{ .movaps, r, q }),
                .mem => |m| try s.b.emit(.sseRM, .{ movFor(tp, true), r, m }),
            }
        }
        /// `r = r op l` (a slot operand is aligned, as packed ops require).
        fn sseOp(s: *Self, op: bld.SseOp, r: Xmm, l: FLoc) !void {
            switch (l) {
                .reg => |q| try s.b.emit(.sseRR, .{ op, r, q }),
                .mem => |m| {
                    s.folded_mem += 1;
                    try s.b.emit(.sseRMop, .{ op, r, m });
                },
            }
        }
        fn pshufd(s: *Self, r: Xmm, l: FLoc, imm: u8) !void {
            switch (l) {
                .reg => |q| try s.b.emit(.sseRRI, .{ .pshufd, r, q, imm }),
                .mem => |m| try s.b.emit(.sseRMI, .{ .pshufd, r, m, imm }),
            }
        }

        // ---- frame ----
        fn epilog(s: *Self) !void {
            if (s.adj != 0) try s.b.emit(.addRI, .{ .rsp, @intCast(s.adj) });
            var i = s.saved.len;
            while (i > 0) {
                i -= 1;
                try s.b.emit(.pop, .{s.saved[i]});
            }
            if (s.use_rbp) try s.b.emit(.pop, .{.rbp});
            try s.b.add(.ret);
        }
        fn params(s: *Self, n: u32) !void {
            var moves: [ir.max_args]Move = undefined;
            var nm: usize = 0;
            for (0..n) |i| {
                const v: ir.VReg = @intCast(i);
                const seg = s.segReg(v);
                if (i < cc.args.len) {
                    const a = cc.args[i];
                    switch (s.res.locs[v]) {
                        .reg => |r| {
                            moves[nm] = .{ .dst = gpOf(r), .src = .{ .reg = a } };
                            nm += 1;
                        },
                        .stack => |k| {
                            try s.b.emit(.movMR, .{ s.gslot(k), a });
                            if (seg) |r| {
                                moves[nm] = .{ .dst = r, .src = .{ .reg = a } };
                                nm += 1;
                            }
                        },
                        .none => {},
                    }
                }
            }
            try s.parMove(moves[0..nm]);
            // Stack parameters: their registers are free once the register moves are done.
            for (@min(n, cc.args.len)..n) |i| {
                const v: ir.VReg = @intCast(i);
                switch (s.res.locs[v]) {
                    .reg => |r| try s.b.emit(.movRM, .{ gpOf(r), cc.stackArg(i) }),
                    .stack => |k| {
                        const r = s.segReg(v) orelse s0;
                        try s.b.emit(.movRM, .{ r, cc.stackArg(i) });
                        try s.b.emit(.movMR, .{ s.gslot(k), r });
                    },
                    .none => {},
                }
            }
        }
        /// Parallel move: register sources first (a destination is written once no pending
        /// move reads it; a cycle saves one destination in r10), then immediates and memory,
        /// whose destinations no longer hold sources.
        fn parMove(s: *Self, moves: []const Move) !void {
            var pend: [ir.max_args]Move = undefined;
            var np: usize = 0;
            for (moves) |m| if (m.src == .reg and m.src.reg != m.dst) {
                pend[np] = m;
                np += 1;
            };
            while (np != 0) {
                var progress = false;
                var i: usize = 0;
                while (i < np) {
                    const m = pend[i];
                    const blocked = for (pend[0..np]) |o| {
                        if (o.src.reg == m.dst) break true;
                    } else false;
                    if (blocked) {
                        i += 1;
                        continue;
                    }
                    try s.b.emit(.movRR, .{ m.dst, m.src.reg });
                    np -= 1;
                    pend[i] = pend[np];
                    progress = true;
                }
                if (!progress) {
                    const d = pend[0].dst;
                    try s.b.emit(.movRR, .{ s0, d });
                    for (pend[0..np]) |*o| if (o.src.reg == d) {
                        o.src = .{ .reg = s0 };
                    };
                }
            }
            for (moves) |m| switch (m.src) {
                .reg => {},
                .imm => |k| try s.materialize(m.dst, k),
                .mem => |mm| try s.b.emit(.movRM, .{ m.dst, mm }),
            };
        }

        // ---- ops ----
        /// `cd`: the constant value `o` defines (`constDef`); `ap`: the plan of an `arith`.
        fn lowerOp(s: *Self, o: *const ir.Op, cd: ?i64, ap: ArithPlan) !void {
            const b = s.b;
            switch (o.*) {
                .@"const" => |c| try s.setConst(c.dst, c.imm),
                .copy => |c| if (s.types[c.dst] != .i64) {
                    try s.fmov(s.fdst(c.dst), s.floc(c.src), s.types[c.dst]);
                    try s.fput(c.dst, s.fdst(c.dst));
                } else if (cd) |k| {
                    try s.setConst(c.dst, k);
                } else {
                    const l = try s.loc(c.src);
                    // A spilled destination without a segment: store the source directly.
                    if (s.res.locs[c.dst] == .stack and s.segReg(c.dst) == null and l == .reg) {
                        try b.emit(.movMR, .{ s.gslot(s.res.locs[c.dst].stack), l.reg });
                    } else {
                        const w = s.dst(c.dst);
                        try s.movTo(w, l);
                        try s.put(c.dst, w);
                    }
                },
                .arith => |c| if (cd) |k| try s.setConst(c.dst, k) else try s.arith(c, ap),
                .load => |c| {
                    const m = x.ptr(try s.base(c.base), c.disp);
                    if (c.ty == .i64) {
                        const w = s.dst(c.dst);
                        try b.emit(.movRM, .{ w, m });
                        try s.put(c.dst, w);
                    } else {
                        const w = s.fdst(c.dst);
                        try b.emit(.sseRM, .{ movFor(c.ty, false), w, m });
                        try s.fput(c.dst, w);
                    }
                },
                .store => |c| {
                    const m = x.ptr(try s.base(c.base), c.disp);
                    if (c.ty != .i64) {
                        try b.emit(.sseMR, .{ movFor(c.ty, false), m, try s.fsrc(c.src, fs0) });
                    } else if (s.k.get(c.src)) |k| if (fitsImm(k)) {
                        s.folded_imm += 1;
                        try b.emit(.movMI, .{ m, @intCast(k) });
                    } else {
                        try b.emit(.movMR, .{ m, try s.src(c.src, s0) });
                    } else try b.emit(.movMR, .{ m, try s.src(c.src, s0) });
                },
                .fconst => |c| {
                    const w = s.fdst(c.dst);
                    if (c.bits == 0) {
                        try b.emit(.sseRR, .{ .xorps, w, w });
                    } else {
                        try b.emit(.movRI, .{ s0, @bitCast(c.bits) });
                        try b.emit(.movqXR, .{ w, s0 });
                    }
                    try s.fput(c.dst, w);
                },
                .typed => |c| try s.typed(c),
                .fcmp => |c| try s.fcmp(c),
                .cvt => |c| try s.cvt(c),
                .splat => |c| try s.splat(c),
                .extract => |c| try s.extract(c),
                .call => |c| try s.call(c),
                .bra, .cbra => unreachable, // removed by lowerBlockParams
                .br => |tg| if (tg != s.next) try b.emit(.jmp, .{s.labels[tg]}),
                .cbr => |c| try s.cbr(c),
                .ret => |v| {
                    if (s.k.get(v)) |k| try s.materialize(.rax, k) else try s.movTo(.rax, try s.loc(v));
                    try s.epilog();
                },
            }
        }

        /// Base register of a load/store (a known address is materialized in r11).
        fn base(s: *Self, v: ir.VReg) !Gp {
            if (s.k.get(v)) |k| {
                try s.materialize(s1, k);
                return s1;
            }
            return s.src(v, s1);
        }

        fn arith(s: *Self, c: anytype, p: ArithPlan) !void {
            const b = s.b;
            switch (c.op) {
                .sdiv, .srem => return s.divide(c, p),
                .shl, .shr, .sar => return s.shift(c, p),
                else => {},
            }
            const w = s.dst(c.dst);
            if (p.a != null and c.op == .sub) {
                // imm - b
                const k = p.a.?;
                const lb = try s.loc(c.b);
                s.folded_imm += 1;
                if (lb == .reg and lb.reg == w) {
                    if (fitsImm(k)) {
                        try b.emit(.neg, .{w});
                        try b.emit(.aluRI, .{ .add, w, @intCast(k) });
                    } else {
                        try b.emit(.movRI, .{ s0, k });
                        try b.emit(.aluRR, .{ .sub, s0, w });
                        try b.emit(.movRR, .{ w, s0 });
                    }
                } else {
                    try s.materialize(w, k);
                    try s.opRL(.sub, w, lb);
                }
                return s.put(c.dst, w);
            }
            if (p.a != null or p.b != null) {
                // `other op imm` (commutative ops with a known left operand are swapped).
                const k = (p.b orelse p.a).?;
                const l = try s.loc(if (p.b != null) c.a else c.b);
                s.folded_imm += 1;
                if (c.op == .mul) {
                    if (pow2Shift(k)) |sh| {
                        try s.movTo(w, l);
                        if (sh != 0) try b.emit(.shiftRI, .{ .shl, w, sh });
                    } else if (l == .reg) {
                        try b.emit(.imulRRI, .{ w, l.reg, @intCast(k) });
                    } else {
                        try s.movTo(w, l);
                        try b.emit(.imulRRI, .{ w, w, @intCast(k) });
                    }
                } else if ((c.op == .add or (c.op == .sub and k != std.math.minInt(i32))) and l == .reg and l.reg != w) {
                    try b.emit(.lea, .{ w, x.ptr(l.reg, @intCast(if (c.op == .add) k else -k)) });
                } else {
                    try s.movTo(w, l);
                    try b.emit(.aluRI, .{ aluOf(c.op), w, @intCast(k) });
                }
                return s.put(c.dst, w);
            }
            const la = try s.loc(c.a);
            const lb = try s.loc(c.b);
            if (la == .reg and la.reg == w) {
                try s.opRL(c.op, w, lb);
            } else if (lb == .reg and lb.reg == w) {
                // The result takes b's register (b dies here).
                if (c.op == .sub) {
                    try b.emit(.neg, .{w});
                    try s.opRL(.add, w, la);
                } else try s.opRL(c.op, w, la);
            } else if (c.op == .add and la == .reg and lb == .reg) {
                try b.emit(.lea, .{ w, x.ptrIdx(la.reg, lb.reg, .x1, 0) });
            } else {
                try s.movTo(w, la);
                try s.opRL(c.op, w, lb);
            }
            try s.put(c.dst, w);
        }

        /// `idiv`: dividend in rax (sign-extended into rdx), divisor in a register other than
        /// rax/rdx (r11 when needed); rax:rdx are clobbered (`regalloc.Config.clobbers`).
        fn divide(s: *Self, c: anytype, p: ArithPlan) !void {
            const b = s.b;
            var divisor: Gp = s1;
            if (p.b) |k| {
                try s.materialize(s1, k);
            } else switch (try s.loc(c.b)) {
                .reg => |r| if (r == .rax or r == .rdx) try b.emit(.movRR, .{ s1, r }) else {
                    divisor = r;
                },
                .mem => |m| try b.emit(.movRM, .{ s1, m }),
            }
            if (p.a) |k| try s.materialize(.rax, k) else try s.movTo(.rax, try s.loc(c.a));
            try b.add(.cqo);
            try b.emit(.idivR, .{divisor});
            try s.put(c.dst, if (c.op == .sdiv) .rax else .rdx);
        }

        /// Shifts: an immediate count is `count & 63`; a variable count goes to cl (rcx is
        /// clobbered, `regalloc.Config.clobbers`).
        fn shift(s: *Self, c: anytype, p: ArithPlan) !void {
            const b = s.b;
            const kind: bld.Shift = switch (c.op) {
                .shl => .shl,
                .shr => .shr,
                else => .sar,
            };
            const w = s.dst(c.dst);
            if (p.b) |k| {
                s.folded_imm += 1;
                try s.movTo(w, try s.loc(c.a));
                const sh: u8 = @intCast(k & 63);
                if (sh != 0) try b.emit(.shiftRI, .{ kind, w, sh });
                return s.put(c.dst, w);
            }
            const lb = try s.loc(c.b);
            const la: ?Loc = if (p.a == null) try s.loc(c.a) else null;
            const a_in_rcx = if (la) |l| l == .reg and l.reg == .rcx else false;
            // Compute in r10 when the result or the left operand occupies rcx.
            const r = if (w == .rcx or a_in_rcx) s0 else w;
            if (r == s0) {
                if (la) |l| try s.movTo(s0, l) else try s.materialize(s0, p.a.?);
                try s.movTo(.rcx, lb);
            } else {
                // rcx first: `w` may hold b (b dies here), never a.
                try s.movTo(.rcx, lb);
                if (la) |l| try s.movTo(w, l) else try s.materialize(w, p.a.?);
            }
            try b.emit(.shiftRCl, .{ kind, r });
            try s.put(c.dst, r);
        }

        fn cbr(s: *Self, c: anytype) !void {
            const b = s.b;
            const p = planCmp(c, &s.k);
            if (p.static) |taken| {
                const tg = if (taken) c.then else c.@"else";
                if (tg != s.next) try b.emit(.jmp, .{s.labels[tg]});
                return;
            }
            var cond = condOf(c.cond);
            if (p.b != null or p.a != null) {
                const k = (p.b orelse p.a).?;
                if (p.b == null) cond = swapCond(cond);
                s.folded_imm += 1;
                const r = try s.src(if (p.b != null) c.a else c.b, s0);
                if (k == 0) try b.emit(.testRR, .{ r, r }) else try b.emit(.cmpRI, .{ r, @intCast(k) });
            } else {
                const la = try s.loc(c.a);
                const lb = try s.loc(c.b);
                if (la == .reg) {
                    switch (lb) {
                        .reg => |q| try b.emit(.cmpRR, .{ la.reg, q }),
                        .mem => |m| {
                            s.folded_mem += 1;
                            try b.emit(.aluRM, .{ .cmp, la.reg, m });
                        },
                    }
                } else if (lb == .reg) {
                    s.folded_mem += 1;
                    try b.emit(.aluRM, .{ .cmp, lb.reg, la.mem });
                    cond = swapCond(cond);
                } else {
                    try b.emit(.movRM, .{ s0, la.mem });
                    try b.emit(.aluRM, .{ .cmp, s0, lb.mem });
                }
            }
            try s.branch(cond, c.then, c.@"else");
        }

        /// Conditional branch by layout: fall through to the next block when possible.
        fn branch(s: *Self, cond: x.Cond, then: u32, els: u32) !void {
            const b = s.b;
            if (then == els) {
                if (then != s.next) try b.emit(.jmp, .{s.labels[then]});
            } else if (then == s.next) {
                try b.emit(.jcc, .{ invertCond(cond), s.labels[els] });
            } else {
                try b.emit(.jcc, .{ cond, s.labels[then] });
                if (els != s.next) try b.emit(.jmp, .{s.labels[els]});
            }
        }

        fn call(s: *Self, c: anytype) !void {
            const b = s.b;
            // Split ranges: save caller-saved registers live across this call.
            const saves = s.call_saves[s.call_start[s.ncall]..s.call_start[s.ncall + 1]];
            s.ncall += 1;
            for (saves) |v| {
                try b.emit(.movMR, .{ s.gslot(s.res.save_slot[v].?), gpOf(s.res.locs[v].reg) });
            }
            // Stack arguments: pushed right to left, padded so rsp stays 16-aligned.
            const nreg = @min(c.args.len, cc.args.len);
            const nstack = c.args.len - nreg;
            const pad: u32 = if (nstack % 2 == 1) 8 else 0;
            if (pad != 0) try b.emit(.subRI, .{ .rsp, @intCast(pad) });
            var i = c.args.len;
            while (i > nreg) {
                i -= 1;
                const a = c.args[i];
                if (s.k.get(a)) |k| {
                    try b.emit(.movRI, .{ s0, k });
                    try b.emit(.push, .{s0});
                } else try b.emit(.push, .{try s.src(a, s0)});
            }
            var moves: [ir.max_args]Move = undefined;
            for (c.args[0..nreg], 0..) |a, j| moves[j] = .{ .dst = cc.args[j], .src = if (s.k.get(a)) |k| .{ .imm = k } else switch (try s.loc(a)) {
                .reg => |r| .{ .reg = r },
                .mem => |m| .{ .mem = m },
            } };
            try s.parMove(moves[0..nreg]);
            if (cc.shadow != 0) try b.emit(.subRI, .{ .rsp, @intCast(cc.shadow) });
            try b.emit(.callAbs, .{c.addr});
            const pop: u32 = cc.shadow + pad + 8 * @as(u32, @intCast(nstack));
            if (pop != 0) try b.emit(.addRI, .{ .rsp, @intCast(pop) });
            var result: Gp = .rax;
            for (saves) |v| {
                const r = gpOf(s.res.locs[v].reg);
                if (r == .rax and result == .rax) {
                    try b.emit(.movRR, .{ s1, .rax });
                    result = s1;
                }
                try b.emit(.movRM, .{ r, s.gslot(s.res.save_slot[v].?) });
            }
            try s.put(c.dst, result);
        }

        // ---- floats and vectors ----
        fn typed(s: *Self, c: anytype) !void {
            const b = s.b;
            const w = s.fdst(c.dst);
            const la = s.floc(c.a);
            const lb = s.floc(c.b);
            if (c.op == .mul and c.ty == .v4i32) return s.mulV4i32(c, w, la, lb);
            if (c.op == .mul and c.ty == .v2i64) return s.mulV2i64(c, la, lb);
            const sop: bld.SseOp = switch (c.ty) {
                .f32, .f64, .v4f32, .v2f64 => fop: {
                    const ops: [6][4]bld.SseOp = .{
                        .{ .addss, .addsd, .addps, .addpd },
                        .{ .subss, .subsd, .subps, .subpd },
                        .{ .mulss, .mulsd, .mulps, .mulpd },
                        .{ .divss, .divsd, .divps, .divpd },
                        // x86 semantics = IR semantics: the second operand when unordered or equal.
                        .{ .minss, .minsd, .minps, .minpd },
                        .{ .maxss, .maxsd, .maxps, .maxpd },
                    };
                    const col: usize = switch (c.ty) {
                        .f32 => 0,
                        .f64 => 1,
                        .v4f32 => 2,
                        else => 3,
                    };
                    break :fop ops[@backingInt(c.op)][col]; // and/or/xor: integer vectors only
                },
                .v4i32, .v2i64 => switch (c.op) {
                    .add => if (c.ty == .v4i32) .paddd else .paddq,
                    .sub => if (c.ty == .v4i32) .psubd else .psubq,
                    .@"and" => .pand,
                    .@"or" => .por,
                    .xor => .pxor,
                    else => unreachable, // float-only ops (rejected by the type check)
                },
                .i64 => unreachable,
            };
            // Commutative up to the NaN payload chosen for two NaN inputs (any NaN equals
            // any NaN in the IR semantics); min/max are not commutative.
            const comm = switch (c.op) {
                .add, .mul, .@"and", .@"or", .xor => true,
                else => false,
            };
            if (la == .reg and la.reg == w) {
                try s.sseOp(sop, w, lb);
            } else if (lb == .reg and lb.reg == w) {
                if (comm) {
                    try s.sseOp(sop, w, la);
                } else {
                    try s.fmov(fs0, la, c.ty);
                    try b.emit(.sseRR, .{ sop, fs0, w });
                    try b.emit(.sseRR, .{ .movaps, w, fs0 });
                }
            } else {
                try s.fmov(w, la, c.ty);
                try s.sseOp(sop, w, lb);
            }
            try s.fput(c.dst, w);
        }

        fn mulV4i32(s: *Self, c: anytype, w: Xmm, la: FLoc, lb: FLoc) !void {
            const b = s.b;
            if (s.sse41) {
                // pmulld xmm, xmm (register operands only).
                if (la == .reg and la.reg == w) {
                    try b.emit(.pmulld, .{ w, try s.fsrc(c.b, fs1) });
                } else if (lb == .reg and lb.reg == w) {
                    try b.emit(.pmulld, .{ w, try s.fsrc(c.a, fs1) });
                } else {
                    try s.fmov(w, la, c.ty);
                    try b.emit(.pmulld, .{ w, try s.fsrc(c.b, fs1) });
                }
                return s.fput(c.dst, w);
            }
            // SSE2: products of the odd lanes (pmuludq on lanes 1/3), of the even lanes, then
            // the low 32 bits of each interleaved back.
            try s.pshufd(fs0, la, 0xF5);
            try s.pshufd(fs1, lb, 0xF5);
            try b.emit(.sseRR, .{ .pmuludq, fs0, fs1 });
            try s.fmov(fs1, la, c.ty);
            try s.sseOp(.pmuludq, fs1, lb);
            try b.emit(.sseRRI, .{ .pshufd, fs1, fs1, 0x08 });
            try b.emit(.sseRRI, .{ .pshufd, fs0, fs0, 0x08 });
            try b.emit(.sseRR, .{ .punpckldq, fs1, fs0 });
            try s.fput(c.dst, fs1);
        }

        /// v2i64 `mul`: no SSE instruction; scalar `imul` per lane.
        fn mulV2i64(s: *Self, c: anytype, la: FLoc, lb: FLoc) !void {
            const b = s.b;
            try s.lane64(s0, la, 0);
            try s.lane64(s1, lb, 0);
            try b.emit(.imulRR, .{ s0, s1 });
            try b.emit(.movqXR, .{ fs1, s0 });
            try s.lane64(s0, la, 1);
            try s.lane64(s1, lb, 1);
            try b.emit(.imulRR, .{ s0, s1 });
            try b.emit(.movqXR, .{ fs0, s0 });
            try b.emit(.sseRR, .{ .punpcklqdq, fs1, fs0 });
            try s.fput(c.dst, fs1);
        }
        /// 64-bit lane `lane` of a vector into `r` (uses fs0 for lane 1 of a register).
        fn lane64(s: *Self, r: Gp, l: FLoc, lane: u8) !void {
            switch (l) {
                .reg => |q| if (lane == 0) try s.b.emit(.movqRX, .{ r, q }) else {
                    try s.b.emit(.sseRRI, .{ .pshufd, fs0, q, 0xEE });
                    try s.b.emit(.movqRX, .{ r, fs0 });
                },
                .mem => |m| try s.b.emit(.movRM, .{ r, x.ptr(m.base.?, m.disp + 8 * @as(i32, lane)) }),
            }
        }

        fn fcmp(s: *Self, c: anytype) !void {
            const b = s.b;
            const ucomi: bld.SseOp = if (c.ty == .f64) .ucomisd else .ucomiss;
            const w = s.dst(c.dst);
            const two = c.cond == .eq or c.cond == .ne;
            // Zero the result registers before the compare (xor clobbers flags).
            try b.emit(.zero32, .{w});
            if (two) try b.emit(.zero32, .{s1});
            // ucomis: unordered sets ZF=PF=CF=1. lt/le swap operands so that `a`/`ae`
            // (CF=0) are false on NaN; eq/ne combine with PF.
            const swap = c.cond == .lt or c.cond == .le;
            const first = try s.fsrc(if (swap) c.b else c.a, fs0);
            try s.sseOp(ucomi, first, s.floc(if (swap) c.a else c.b));
            switch (c.cond) {
                .eq, .ne => {
                    try b.emit(.setcc, .{ if (c.cond == .eq) .e else .ne, w });
                    try b.emit(.setcc, .{ if (c.cond == .eq) .np else .p, s1 });
                    try b.emit(.aluRR, .{ if (c.cond == .eq) .@"and" else .@"or", w, s1 });
                },
                .lt, .gt => try b.emit(.setcc, .{ .a, w }),
                .le, .ge => try b.emit(.setcc, .{ .ae, w }),
            }
            try s.put(c.dst, w);
        }

        fn cvt(s: *Self, c: anytype) !void {
            const b = s.b;
            if (c.to == .i64) {
                const w = s.dst(c.dst);
                try b.emit(.cvtToInt, .{ if (c.from == .f64) .cvttsd2si else .cvttss2si, w, try s.fsrc(c.src, fs0) });
                return s.put(c.dst, w);
            }
            const w = s.fdst(c.dst);
            if (c.from == .i64) {
                const r = if (s.k.get(c.src)) |k| blk: {
                    try s.materialize(s0, k);
                    break :blk s0;
                } else try s.src(c.src, s0);
                try b.emit(.sseRR, .{ .xorps, w, w }); // break the false dependency
                try b.emit(.cvtFromInt, .{ if (c.to == .f64) .cvtsi2sd else .cvtsi2ss, w, r });
            } else {
                try s.sseOp(if (c.to == .f64) .cvtss2sd else .cvtsd2ss, w, s.floc(c.src));
            }
            try s.fput(c.dst, w);
        }

        fn splat(s: *Self, c: anytype) !void {
            const b = s.b;
            const w = s.fdst(c.dst);
            switch (c.ty) {
                .v4i32, .v2i64 => {
                    const r = if (s.k.get(c.src)) |k| blk: {
                        try s.materialize(s0, k);
                        break :blk s0;
                    } else try s.src(c.src, s0);
                    if (c.ty == .v4i32) {
                        try b.emit(.movdXR, .{ w, r }); // truncates to the low 32 bits
                        try b.emit(.sseRRI, .{ .pshufd, w, w, 0 });
                    } else {
                        try b.emit(.movqXR, .{ w, r });
                        try b.emit(.sseRR, .{ .punpcklqdq, w, w });
                    }
                },
                .v4f32 => {
                    try s.fmov(w, s.floc(c.src), .f32);
                    try b.emit(.sseRRI, .{ .shufps, w, w, 0 });
                },
                else => {
                    try s.fmov(w, s.floc(c.src), .f64);
                    try b.emit(.sseRR, .{ .unpcklpd, w, w });
                },
            }
            try s.fput(c.dst, w);
        }

        fn extract(s: *Self, c: anytype) !void {
            const b = s.b;
            const l = s.floc(c.src);
            const lane: i32 = c.lane;
            switch (c.ty) {
                .v4i32, .v2i64 => {
                    const w = s.dst(c.dst);
                    if (c.ty == .v2i64) {
                        try s.lane64(w, l, c.lane);
                    } else {
                        const v = switch (l) {
                            .reg => |q| q,
                            .mem => |m| blk: {
                                try b.emit(.sseRM, .{ .movaps, fs0, m });
                                break :blk fs0;
                            },
                        };
                        if (c.lane == 0) {
                            try b.emit(.movdRX, .{ w, v });
                        } else {
                            try b.emit(.sseRRI, .{ .pshufd, fs0, v, c.lane });
                            try b.emit(.movdRX, .{ w, fs0 });
                        }
                        try b.emit(.movsxd, .{ w, w }); // lanes are signed
                    }
                    try s.put(c.dst, w);
                },
                else => {
                    const w = s.fdst(c.dst);
                    const f32_lanes = c.ty == .v4f32;
                    switch (l) {
                        .reg => |q| if (c.lane == 0) {
                            if (q != w) try b.emit(.sseRR, .{ .movaps, w, q });
                        } else try b.emit(.sseRRI, .{ .pshufd, w, q, if (f32_lanes) c.lane else 0xEE }),
                        .mem => |m| try b.emit(.sseRM, .{ if (f32_lanes) .movss else .movsd, w, x.ptr(m.base.?, m.disp + lane * @as(i32, if (f32_lanes) 4 else 8)) }),
                    }
                    try s.fput(c.dst, w);
                },
            }
        }
    };
}

/// IR -> bytes in `cb` (finalized). Runs dead-code, jump-to-next and branch-shortening passes.
pub fn compile(gpa: std.mem.Allocator, f: *const ir.Func, cb: *CodeBuffer, opts: Options) !Stats {
    if (opts.shorten_branches) {
        var stream = @import("compiler_buffer.zig").Buffer.init(gpa, cb);
        defer stream.deinit();
        var st = try lowerInto(gpa, f, &stream, opts);
        try stream.finish();
        try cb.finalize();
        st.nodes = stream.count();
        return st;
    }
    var b = bld.Builder.init(gpa);
    defer b.deinit();
    var st = try lower(gpa, f, &b, opts);
    try b.runPasses(.{ bld.removeDeadCode, bld.removeJumpsToNext });
    // The node list is complete: drop its growth slack before the output buffer grows.
    b.nodes.shrinkAndFree(gpa, b.nodes.items.len);
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
    // A leaf without spills or callee-saved registers has no frame; the loop is one compare at
    // the top, an inverted rel8 exit branch and a rel8 back edge.
    if (!win) try t.expectEqualSlices(u8, &.{
        0x31, 0xC0, // xor eax, eax
        0x41, 0xB8, 0x01, 0x00, 0x00, 0x00, // mov r8d, 1
        0x45, 0x31, 0xC9, // xor r9d, r9d
        0x49, 0x39, 0xF9, // loop: cmp r9, rdi
        0x7D, 0x10, // jge exit
        0x4A, 0x8D, 0x34, 0x00, // lea rsi, [rax+r8]
        0x4C, 0x89, 0xC0, // mov rax, r8
        0x49, 0x89, 0xF0, // mov r8, rsi
        0x49, 0x83, 0xC1, 0x01, // add r9, 1
        0xEB, 0xEB, // jmp loop
        0xC3, // exit: ret
    }, c);
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
    // add 5 (lea), imul 7, cmp 7 and `sub v1` in block a (v1 is a block-0 constant) fold; 3e9
    // does not fit imm32. v1 and v4 never need a register.
    try t.expectEqual(@as(u32, 4), st.folded_imm);
    try t.expectEqual(@as(u32, 2), st.excluded);
    if (!win) try t.expectEqualSlices(u8, &.{
        0x48, 0x8D, 0x47, 0x05, // lea rax, [rdi+5]
        0x41, 0xB8, 0x00, 0x5E, 0xD0, 0xB2, // mov r8d, 3000000000
        0x49, 0x0F, 0xAF, 0xC0, // imul rax, r8
        0x48, 0x6B, 0xC0, 0x07, // imul rax, rax, 7
        0x48, 0x83, 0xF8, 0x07, // cmp rax, 7
        0x7D, 0x05, // jge b
        0x48, 0x83, 0xE8, 0x05, // a: sub rax, 5
        0xC3, // ret
        0xC3, // b: ret
    }, cb.code());

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
