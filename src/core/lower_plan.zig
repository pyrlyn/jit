//! Arch-agnostic pre-pass shared by the IR compilers (see docs/ROADMAP.md, "codegen quality").
//!
//! 1. Block layout with loop rotation: a `br H` whose target `H` holds nothing but a `cbr` is
//!    emitted as a copy of that `cbr` (tail duplication), so a `while` loop tests its condition
//!    at the bottom and takes one branch per iteration. Blocks no longer reached are not emitted;
//!    `next[b]` is the block laid out after `b` (fallthrough), so `br next` and the `else` jump of
//!    a `cbr` disappear and a `cbr` whose `then` is next is inverted.
//!    Register allocation still sees the original CFG: `H` defines nothing, so every value has the
//!    same location at the end of the predecessor as at the end of `H` (one location per vreg).
//! 2. Constants: `konst[v]` is set for a vreg with exactly one definition, a `const` in the entry
//!    block (which is never a branch target), and no use in the entry block before that
//!    definition, so the value is the same at every use. The compiler folds such operands into
//!    immediates where `foldable` says the target can encode them; a `const` all of whose uses are
//!    folded is dropped (`skip_const`), together with its register.
//! 3. Shift fusion: `X = shl A, K` (K constant) immediately followed by `Y = add B, X` (either
//!    order) where `X` is not used anywhere else (or `X == Y`) becomes one `add Y, B, A, lsl K`
//!    (`fuse[add] = .{A, K}`, `kind[shl] = .skip`).
const std = @import("std");
const ir = @import("ir.zig");

pub const Target = enum { a64, x64 };

pub const Kind = enum(u8) { normal, skip, fused };

/// Per-op flags; ops are numbered globally, `base[bi] + i`.
pub const Plan = struct {
    base: []u32,
    kind: []Kind,
    /// Bit 0: operand `a` folded to an immediate, bit 1: operand `b` (arith and cbr only).
    folded: []u2,
    /// Fused add: shifted operand `a_src` and shift amount.
    fuse_src: []ir.VReg,
    fuse_sh: []u6,
    konst: []?i64,
    skip_const: []bool,
    /// Tail-duplicated condition block emitted at the end of `b` (instead of `br`).
    dup: []?u32,
    emitted: []bool,
    /// Next emitted block after `b` in layout order (null: the epilog follows).
    next: []?u32,
    /// Function contains a call.
    has_calls: bool,
    /// Constants folded into instructions / ops removed by fusion (stats).
    n_folded: u32 = 0,
    n_fused: u32 = 0,
    n_rotated: u32 = 0,

    pub fn idx(p: Plan, bi: usize, i: usize) usize {
        return p.base[bi] + i;
    }
};

fn isCondOnly(f: *const ir.Func, b: u32) bool {
    const ops = f.blocks.items[b].ops.items;
    return ops.len == 1 and ops[0] == .cbr;
}

/// Whether constant `k` can be encoded in place of operand `which` (0 = a, 1 = b) of `o`.
fn foldable(comptime target: Target, o: ir.Op, which: u1, k: i64) bool {
    switch (target) {
        .a64 => switch (o) {
            .arith => |c| return switch (c.op) {
                .add => k >= -4095 and k <= 4095,
                .sub => which == 1 and k >= -4095 and k <= 4095,
                .shl, .shr, .sar => which == 1,
                else => false,
            },
            .cbr => return k >= 0 and k <= 4095,
            else => return false,
        },
        .x64 => return false, // x86 keeps its block-local folding (compiler.zig)
    }
}

pub fn build(A: std.mem.Allocator, f: *const ir.Func, comptime target: Target, optimize: bool) !Plan {
    const nb = f.blocks.items.len;
    const nv: usize = @max(f.num_vregs, f.params, 1);
    var total: u32 = 0;
    const base = try A.alloc(u32, nb);
    for (f.blocks.items, 0..) |blk, bi| {
        base[bi] = total;
        total += @intCast(blk.ops.items.len);
    }
    var p: Plan = .{
        .base = base,
        .kind = try A.alloc(Kind, total),
        .folded = try A.alloc(u2, total),
        .fuse_src = try A.alloc(ir.VReg, total),
        .fuse_sh = try A.alloc(u6, total),
        .konst = try A.alloc(?i64, nv),
        .skip_const = try A.alloc(bool, nv),
        .dup = try A.alloc(?u32, nb),
        .emitted = try A.alloc(bool, nb),
        .next = try A.alloc(?u32, nb),
        .has_calls = false,
    };
    @memset(p.kind, .normal);
    @memset(p.folded, 0);
    @memset(p.konst, null);
    @memset(p.skip_const, false);
    @memset(p.dup, null);
    for (f.blocks.items) |blk| for (blk.ops.items) |o| if (o == .call) {
        p.has_calls = true;
    };
    if (!optimize) {
        @memset(p.emitted, true);
        for (0..nb) |b| p.next[b] = null; // keep every branch
        return p;
    }

    // ---- constants ----
    const defs = try A.alloc(u32, nv);
    const uses = try A.alloc(u32, nv);
    const folded_uses = try A.alloc(u32, nv);
    @memset(defs, 0);
    @memset(uses, 0);
    @memset(folded_uses, 0);
    for (0..f.params) |v| defs[v] = 1;
    var entry_targeted = false;
    for (f.blocks.items) |blk| for (blk.ops.items) |o| {
        var buf: [ir.max_args]ir.VReg = undefined;
        for (o.uses(&buf)) |u| uses[u] += 1;
        if (o.def()) |d| defs[d] += 1;
        switch (o) {
            .br => |tg| entry_targeted = entry_targeted or tg == 0,
            .cbr => |c| entry_targeted = entry_targeted or c.then == 0 or c.@"else" == 0,
            .bra => |c| entry_targeted = entry_targeted or c.target == 0,
            .cbra => |c| entry_targeted = entry_targeted or c.then == 0 or c.@"else" == 0,
            else => {},
        }
    };
    if (!entry_targeted and nb > 0) {
        const seen_def = try A.alloc(bool, nv);
        const early_use = try A.alloc(bool, nv);
        @memset(seen_def, false);
        @memset(early_use, false);
        for (f.blocks.items[0].ops.items) |o| {
            var buf: [ir.max_args]ir.VReg = undefined;
            for (o.uses(&buf)) |u| if (!seen_def[u]) {
                early_use[u] = true;
            };
            if (o == .@"const") {
                const c = o.@"const";
                if (defs[c.dst] == 1 and !early_use[c.dst] and c.dst >= f.params) p.konst[c.dst] = c.imm;
            }
            if (o.def()) |d| seen_def[d] = true;
        }
    }

    // ---- folding and fusion ----
    for (f.blocks.items, 0..) |blk, bi| {
        const ops = blk.ops.items;
        for (ops, 0..) |o, i| {
            const g = base[bi] + i;
            if (p.kind[g] != .normal) continue;
            switch (o) {
                .arith => |c| {
                    // shl-by-constant feeding the next add
                    if (c.op == .shl and i + 1 < ops.len and ops[i + 1] == .arith and target == .a64) if (p.konst[c.b]) |k| {
                        const n = ops[i + 1].arith;
                        const x = c.dst;
                        if (n.op == .add and (n.a == x) != (n.b == x) and (n.dst == x or uses[x] == 1)) {
                            p.kind[g] = .skip;
                            p.kind[g + 1] = .fused;
                            p.fuse_src[g + 1] = c.a;
                            p.fuse_sh[g + 1] = @truncate(@as(u64, @bitCast(k)));
                            folded_uses[c.b] += 1;
                            p.n_fused += 1;
                            continue;
                        }
                    };
                    if (p.konst[c.b]) |k| if (foldable(target, o, 1, k)) {
                        p.folded[g] = 2;
                        folded_uses[c.b] += 1;
                        continue;
                    };
                    if (p.konst[c.a]) |k| if (foldable(target, o, 0, k)) {
                        p.folded[g] = 1;
                        folded_uses[c.a] += 1;
                    };
                },
                .cbr => |c| {
                    if (p.konst[c.b]) |k| if (foldable(target, o, 1, k)) {
                        p.folded[g] = 2;
                        folded_uses[c.b] += 1;
                        continue;
                    };
                    if (p.konst[c.a]) |k| if (foldable(target, o, 0, k)) {
                        p.folded[g] = 1;
                        folded_uses[c.a] += 1;
                    };
                },
                else => {},
            }
        }
    }
    for (0..nv) |v| if (p.konst[v] != null and folded_uses[v] == uses[v]) {
        p.skip_const[v] = true;
        p.n_folded += folded_uses[v];
    };

    // ---- layout ----
    for (f.blocks.items, 0..) |blk, bi| {
        const ops = blk.ops.items;
        if (ops.len == 0) continue;
        if (ops[ops.len - 1] == .br) {
            const tg = ops[ops.len - 1].br;
            if (tg != bi and isCondOnly(f, tg)) {
                p.dup[bi] = tg;
                p.n_rotated += 1;
            }
        }
    }
    @memset(p.emitted, false);
    var stack: std.ArrayList(u32) = .empty;
    try stack.append(A, 0);
    while (stack.pop()) |b| {
        if (p.emitted[b]) continue;
        p.emitted[b] = true;
        const ops = f.blocks.items[b].ops.items;
        if (ops.len == 0) continue;
        var t: ?ir.Op = ops[ops.len - 1];
        if (p.dup[b]) |h| t = f.blocks.items[h].ops.items[0];
        switch (t.?) {
            .br => |tg| try stack.append(A, tg),
            .cbr => |c| {
                try stack.append(A, c.@"else");
                try stack.append(A, c.then);
            },
            else => {},
        }
    }
    var last: ?u32 = null;
    var bi: usize = nb;
    while (bi > 0) {
        bi -= 1;
        p.next[bi] = last;
        if (p.emitted[bi]) last = @intCast(bi);
    }
    return p;
}
