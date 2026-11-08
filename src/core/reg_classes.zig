//! Register classes for the compilers: integer (`i64`) values live in GP registers, scalar
//! float (`f32`/`f64`) values in FP/vector registers. Each class is allocated by its own
//! `regalloc` run over the same instruction positions (the entry block's parameter-def
//! instruction first, then one instruction per IR op), so positions, call sites and segments
//! line up between the two results. Vregs of the other class are simply absent from a run.
//!
//! FP values are never passed in registers across the ABI (IR calls and `ret` are i64-only),
//! so the FP run uses `callee_saved = 0`: a float value live across a call is spilled.
const std = @import("std");
const ir = @import("ir.zig");
const ra = @import("regalloc.zig");

pub const Class = enum { gp, fp };

pub fn classOf(tp: ir.Type) Class {
    return if (tp == .f32 or tp == .f64) .fp else .gp;
}

/// Float IR features the register-allocating compilers lower (x86-64 and AArch64).
/// Vector types, `splat` and `extract` are still rejected (`error.UnsupportedType`).
pub const Support = struct {
    f32: bool = true,
    f64: bool = true,
    /// `cvt` (i64 <-> f32/f64, f32 <-> f64).
    cvt: bool = true,
};
pub const support: Support = .{};

/// Checks that `f` only uses float features in `support`; returns vreg types (owned by `gpa`).
pub fn check(gpa: std.mem.Allocator, f: *const ir.Func) ![]ir.Type {
    const ty = f.typeCheck(gpa) catch |e| return switch (e) {
        error.TypeMismatch => error.TypeMismatch,
        error.OutOfMemory => error.OutOfMemory,
    };
    errdefer gpa.free(ty);
    const okType = struct {
        fn f_(tp: ir.Type) bool {
            return switch (tp) {
                .i64 => true,
                .f32 => support.f32,
                .f64 => support.f64,
                else => false,
            };
        }
    }.f_;
    for (ty) |tp| if (!okType(tp)) return error.UnsupportedType;
    // `typeCheck` ties every op's type to a vreg type checked above: `splat` defines and
    // `extract` reads a vector vreg, `fconst`/`typed`/`load` define one of `ty`, `fcmp`/
    // `store` read one, `cvt` defines `to` and reads `from`. Only a disabled `cvt` needs a
    // pass over the ops.
    if (!support.cvt) for (f.blocks.items) |b| for (b.ops.items) |*o| if (o.* == .cvt) return error.UnsupportedType;
    return ty;
}

/// Allocation input for one class (`A` should be an arena; nothing is freed individually).
/// `types = null`: every vreg is GP (integer-only compilers). Instructions, vreg lists and
/// successors share one allocation, sized before filling the arrays. The compiler uses
/// compact operand offsets; this entry point preserves the public slice-based input.
pub fn buildFunc(A: std.mem.Allocator, f: *const ir.Func, types: ?[]const ir.Type, class: Class, mark_copies: bool) !ra.Func {
    return buildInput(false, A, f, types, class, mark_copies);
}

fn buildInput(comptime compact: bool, A: std.mem.Allocator, f: *const ir.Func, types: ?[]const ir.Type, class: Class, mark_copies: bool) !if (compact) ra.CompactFunc else ra.Func {
    // Parameters are i64 (GP); the FP run gets an empty placeholder at the same position.
    const nparams: usize = if (class == .gp) f.params else 0;
    var ninst: usize = 1;
    var nsucc: usize = 0;
    var nvars: usize = nparams;
    // A branch back to the entry block: the parameter defs get a block of their own (same
    // positions), or they would kill the parameters at that back edge in liveness.
    var entry_is_target = false;
    for (f.blocks.items) |blk| switch (blk.ops.items[blk.ops.items.len - 1]) {
        .br => |tg| entry_is_target = entry_is_target or tg == 0,
        .cbr => |c| entry_is_target = entry_is_target or c.then == 0 or c.@"else" == 0,
        else => {},
    };
    const pre: usize = @intFromBool(entry_is_target);
    for (f.blocks.items) |blk| {
        ninst += blk.ops.items.len;
        for (blk.ops.items) |o| nvars += switch (o) {
            .call => |c| c.args.len + 1,
            .bra => |c| c.args.len,
            .cbra => |c| 2 + c.then_args.len + c.else_args.len,
            else => 3,
        };
        nsucc += switch (blk.ops.items[blk.ops.items.len - 1]) {
            .br => 1,
            .cbr => 2,
            else => 0,
        };
    }
    const B = if (compact) ra.CompactBlock else ra.Block;
    const I = if (compact) ra.CompactInst else ra.Inst;
    // One arena allocation, with no abandoned buffers from ArrayList/arena growth.
    const bytes = (f.blocks.items.len + pre) * @sizeOf(B) + ninst * @sizeOf(I) +
        (nsucc + pre + nvars) * @sizeOf(u32);
    const storage = try A.alloc(usize, std.mem.alignForward(usize, bytes, @sizeOf(usize)) / @sizeOf(usize));
    var fixed = std.heap.FixedBufferAllocator.init(std.mem.sliceAsBytes(storage));
    const tmp = fixed.allocator();
    const blocks = try tmp.alloc(B, f.blocks.items.len + pre);
    const all_insts = try tmp.alloc(I, ninst);
    const succ_buf = try tmp.alloc(u32, nsucc + pre);
    var vs: std.ArrayList(ra.VReg) = .empty;
    try vs.ensureTotalCapacityPrecise(tmp, nvars);
    var ii: usize = 0;
    var si: usize = 0;
    for (0..nparams) |i| vs.appendAssumeCapacity(@intCast(i));
    // Slices get their lengths now and their pointers into `vs` at the end.
    const unset: [*]const ra.VReg = @ptrFromInt(@alignOf(ra.VReg));
    all_insts[0] = if (compact) .{ .offset = 0, .nuses = 0, .ndefs = @intCast(nparams) } else .{ .defs = unset[0..nparams] };
    if (pre == 1) {
        succ_buf[0] = 1;
        si = 1;
        blocks[0] = .{ .insts = all_insts[0..1], .succs = succ_buf[0..1] };
        ii = 1;
    }
    for (f.blocks.items, 0..) |blk, bi| {
        // Without a pre-entry block, the parameter defs open block 0.
        const extra: usize = if (bi == 0 and pre == 0) 1 else 0;
        const insts = all_insts[ii..][0 .. blk.ops.items.len + extra];
        ii += insts.len;
        for (blk.ops.items, 0..) |*o, i| {
            // One switch for uses (in `Op.uses` order) and the def.
            var p: Push = .{ .vs = vs.allocatedSlice(), .vi = vs.items.len, .types = types, .class = class };
            const ub = vs.items.len;
            var def: ?ir.VReg = null;
            switch (o.*) {
                .br => {},
                .ret => |v| p.add(v),
                inline .@"const", .fconst => |x| def = x.dst,
                .copy => |x| {
                    p.add(x.src);
                    def = x.dst;
                },
                inline .cvt, .splat, .extract => |x| {
                    p.add(x.src);
                    def = x.dst;
                },
                .load => |x| {
                    p.add(x.base);
                    def = x.dst;
                },
                .store => |x| {
                    p.add(x.base);
                    p.add(x.src);
                },
                .cbr => |x| {
                    p.add(x.a);
                    p.add(x.b);
                },
                inline .arith, .typed, .fcmp => |x| {
                    p.add(x.a);
                    p.add(x.b);
                    def = x.dst;
                },
                .call => |x| {
                    for (x.args) |v| p.add(v);
                    def = x.dst;
                },
                .bra => |x| for (x.args) |v| p.add(v),
                .cbra => |x| {
                    p.add(x.a);
                    p.add(x.b);
                    for (x.then_args) |v| p.add(v);
                    for (x.else_args) |v| p.add(v);
                },
            }
            const d0 = p.vi;
            if (def) |dv| p.add(dv);
            vs.items.len = p.vi;
            const nd = p.vi - d0;
            insts[i + extra] = if (compact) .{ .offset = @intCast(ub), .nuses = @intCast(d0 - ub), .ndefs = @intCast(nd), .is_call = o.* == .call, .is_copy = mark_copies and o.* == .copy and nd == 1 } else .{ .uses = unset[0 .. d0 - ub], .defs = unset[0..nd], .is_call = o.* == .call, .is_copy = mark_copies and o.* == .copy and nd == 1 };
        }
        const s0 = si;
        switch (blk.ops.items[blk.ops.items.len - 1]) {
            .br => |tg| {
                succ_buf[si] = @intCast(tg + pre);
                si += 1;
            },
            .cbr => |c| {
                succ_buf[si] = @intCast(c.then + pre);
                succ_buf[si + 1] = @intCast(c.@"else" + pre);
                si += 2;
            },
            else => {},
        }
        blocks[bi + pre] = .{ .insts = insts, .succs = succ_buf[s0..si] };
    }
    if (compact) return .{ .blocks = blocks, .operands = vs.items, .num_vregs = @max(f.num_vregs, 1) };
    var off: usize = 0;
    for (all_insts) |*in| {
        in.uses = vs.items[off..][0..in.uses.len];
        off += in.uses.len;
        in.defs = vs.items[off..][0..in.defs.len];
        off += in.defs.len;
    }
    return .{ .blocks = blocks, .num_vregs = @max(f.num_vregs, 1) };
}

const Push = struct {
    vs: []ra.VReg,
    vi: usize,
    types: ?[]const ir.Type,
    class: Class,
    inline fn add(p: *Push, v: ir.VReg) void {
        const k = if (p.types) |t_| classOf(t_[v]) == p.class else p.class == .gp;
        if (k) {
            p.vs[p.vi] = v;
            p.vi += 1;
        }
    }
};

/// One register-allocation run for `class`: the input is built in a temporary arena that is
/// freed before returning (so the inputs of the two classes are never alive at once, nor
/// during code emission), and the result is checked with `ra.verify` when runtime safety is
/// on. Without any FP vreg the FP run is skipped: every location is `.none`, no slots.
pub fn allocateClass(gpa: std.mem.Allocator, f: *const ir.Func, types: ?[]const ir.Type, class: Class, mark_copies: bool, cfg: ra.Config) !ra.Result {
    if (class == .fp and (types == null or !hasFp(types.?))) return emptyResult(gpa, @max(f.num_vregs, 1));
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const rf = try buildInput(true, arena.allocator(), f, types, class, mark_copies);
    const res = try ra.allocateCompact(gpa, rf, cfg);
    if (std.debug.runtime_safety) std.debug.assert(ra.verifyCompact(rf, cfg, res));
    return res;
}

fn emptyResult(gpa: std.mem.Allocator, n: u32) !ra.Result {
    const locs = try gpa.alloc(ra.Loc, n);
    errdefer gpa.free(locs);
    @memset(locs, .none);
    const iv = try gpa.alloc(ra.Interval, n);
    errdefer gpa.free(iv);
    for (iv, 0..) |*x, v| x.* = .{ .vreg = @intCast(v), .start = std.math.maxInt(u32), .end = std.math.maxInt(u32) };
    const save = try gpa.alloc(?u32, n);
    @memset(save, null);
    return .{ .locs = locs, .intervals = iv, .num_slots = 0, .spills = 0, .save_slot = save };
}

/// True if `f` has any FP-class vreg.
pub fn hasFp(types: []const ir.Type) bool {
    for (types) |tp| if (classOf(tp) == .fp) return true;
    return false;
}

test "reg classes: split GP/FP inputs at equal positions" {
    const t = std.testing;
    var f = try ir.parse(t.allocator,
        \\func f 1
        \\e:
        \\  v1 = fconst.f32 1.5
        \\  v2 = add.f32 v1 v1
        \\  v3 = fcmp.f32 lt v1 v2
        \\  v4 = add v3 v0
        \\  ret v4
    );
    defer f.deinit();
    const ty = try check(t.allocator, &f);
    defer t.allocator.free(ty);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const g = try buildFunc(arena.allocator(), &f, ty, .gp, true);
    const p = try buildFunc(arena.allocator(), &f, ty, .fp, true);
    try t.expectEqual(g.blocks[0].insts.len, p.blocks[0].insts.len);
    try t.expectEqualSlices(ra.VReg, &.{1}, p.blocks[0].insts[2].uses[0..1]); // add.f32 uses v1
    try t.expectEqual(@as(usize, 0), g.blocks[0].insts[2].uses.len);
    try t.expectEqualSlices(ra.VReg, &.{3}, g.blocks[0].insts[3].defs); // fcmp defines a GP value
    try t.expectEqual(@as(usize, 0), p.blocks[0].insts[3].defs.len);
}

test "reg classes: vector values and ops are rejected" {
    const t = std.testing;
    const srcs = [_][]const u8{
        "func f 1\ne:\n  v1 = splat.v2i64 v0\n  ret v0\n",
        "func f 1\ne:\n  v1 = splat.v2i64 v0\n  v2 = extract.v2i64 v1 0\n  ret v2\n",
        "func f 1\ne:\n  v1 = load.v4f32 v0 0\n  ret v0\n",
        "func f 1\ne:\n  v1 = splat.v4i32 v0\n  store.v4i32 v0 0 v1\n  ret v0\n",
        "func f 1\ne:\n  v1 = splat.v2i64 v0\n  v2 = add.v2i64 v1 v1\n  ret v0\n",
    };
    for (srcs) |src| {
        var f = try ir.parse(t.allocator, src);
        defer f.deinit();
        try t.expectError(error.UnsupportedType, check(t.allocator, &f));
    }
}

test "reg classes: compact inputs preserve allocation, calls and split positions" {
    const t = std.testing;
    var rnd = std.Random.DefaultPrng.init(0x20_2026);
    const random = rnd.random();
    for (0..40) |_| {
        var f = ir.Func.init(t.allocator);
        defer f.deinit();
        f.params = ir.max_args;
        for (0..4) |_| _ = try f.addBlock("b");
        const args = try f.alloc().alloc(ir.VReg, ir.max_args);
        for (args, 0..) |*v, i| v.* = @intCast(i);
        for (0..4) |bi| {
            const b: u32 = @intCast(bi);
            for (0..20) |_| {
                const d = random.uintLessThan(u32, ir.max_args);
                const a = random.uintLessThan(u32, ir.max_args);
                switch (random.uintLessThan(u8, 4)) {
                    0 => try f.emit(b, .{ .copy = .{ .dst = d, .src = a } }),
                    1 => try f.emit(b, .{ .call = .{ .dst = d, .addr = 0x1234, .args = args } }),
                    2 => try f.emit(b, .{ .cvt = .{ .dst = ir.max_args, .src = a, .to = .f64, .from = .i64 } }),
                    else => try f.emit(b, .{ .cvt = .{ .dst = d, .src = ir.max_args, .to = .i64, .from = .f64 } }),
                }
            }
            try f.emit(b, .{ .cbr = .{ .cond = .ne, .a = 0, .b = 1, .then = random.uintLessThan(u32, 4), .@"else" = 0 } });
        }
        try f.verify();
        const ty = try check(t.allocator, &f);
        defer t.allocator.free(ty);
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        for ([_]Class{ .gp, .fp }) |class| {
            const plain = try buildFunc(arena.allocator(), &f, ty, class, true);
            const compact = try buildInput(true, arena.allocator(), &f, ty, class, true);
            try t.expectEqual(plain.blocks.len, compact.blocks.len);
            for (plain.blocks, compact.blocks) |a, b| {
                try t.expectEqualSlices(u32, a.succs, b.succs);
                try t.expectEqual(a.insts.len, b.insts.len);
                for (a.insts, b.insts) |x, y| {
                    try t.expectEqualSlices(u32, x.uses, compact.operands[y.offset..][0..y.nuses]);
                    try t.expectEqualSlices(u32, x.defs, compact.operands[@as(usize, y.offset) + y.nuses ..][0..y.ndefs]);
                    try t.expectEqual(x.is_call, y.is_call);
                    try t.expectEqual(x.is_copy, y.is_copy);
                }
            }
            for ([_]ra.Algo{ .linear, .graph }) |algo| for ([_]bool{ false, true }) |split| {
                const cfg: ra.Config = .{ .allocatable = 0b111, .callee_saved = 0b100, .algo = algo, .split = split, .split_at_calls = split };
                var a = try ra.allocate(t.allocator, plain, cfg);
                defer a.deinit(t.allocator);
                var b = try ra.allocateCompact(t.allocator, compact, cfg);
                defer b.deinit(t.allocator);
                try t.expectEqualDeep(a, b);
                try t.expect(ra.verify(plain, cfg, b));
            };
        }
    }
}
