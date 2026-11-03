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
    for (f.blocks.items) |b| for (b.ops.items) |o| switch (o) {
        .splat, .extract => return error.UnsupportedType,
        .cvt => |x| if (!support.cvt or !okType(x.to) or !okType(x.from)) return error.UnsupportedType,
        inline .fconst, .typed, .fcmp => |x| if (!okType(x.ty)) return error.UnsupportedType,
        inline .load, .store => |x| if (!okType(x.ty)) return error.UnsupportedType,
        else => {},
    };
    return ty;
}

/// Allocation input for one class (`A` should be an arena; nothing is freed individually).
pub fn buildFunc(A: std.mem.Allocator, f: *const ir.Func, types: []const ir.Type, class: Class, mark_copies: bool) !ra.Func {
    const blocks = try A.alloc(ra.Block, f.blocks.items.len);
    const Filter = struct {
        fn run(a: std.mem.Allocator, vs: []const ir.VReg, ty: []const ir.Type, c: Class) ![]const ra.VReg {
            var n: usize = 0;
            for (vs) |v| if (classOf(ty[v]) == c) {
                n += 1;
            };
            const out = try a.alloc(ra.VReg, n);
            var i: usize = 0;
            for (vs) |v| if (classOf(ty[v]) == c) {
                out[i] = v;
                i += 1;
            };
            return out;
        }
    };
    for (f.blocks.items, 0..) |blk, bi| {
        const extra: usize = if (bi == 0) 1 else 0;
        const insts = try A.alloc(ra.Inst, blk.ops.items.len + extra);
        if (bi == 0) {
            // Parameters are i64 (GP); the FP run gets an empty placeholder at the same position.
            const params = try A.alloc(ra.VReg, if (class == .gp) f.params else 0);
            for (params, 0..) |*p, i| p.* = @intCast(i);
            insts[0] = .{ .defs = params };
        }
        for (blk.ops.items, 0..) |o, i| {
            var buf: [ir.max_args]ir.VReg = undefined;
            const u = try Filter.run(A, o.uses(&buf), types, class);
            const d: []const ra.VReg = if (o.def()) |dv| try Filter.run(A, &.{dv}, types, class) else &.{};
            insts[i + extra] = .{ .uses = u, .defs = d, .is_call = o == .call, .is_copy = mark_copies and o == .copy and d.len == 1 };
        }
        const succs: []const u32 = switch (blk.ops.items[blk.ops.items.len - 1]) {
            .br => |tg| try A.dupe(u32, &.{tg}),
            .cbr => |c| try A.dupe(u32, &.{ c.then, c.@"else" }),
            else => &.{},
        };
        blocks[bi] = .{ .insts = insts, .succs = succs };
    }
    return .{ .blocks = blocks, .num_vregs = @max(f.num_vregs, 1) };
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
