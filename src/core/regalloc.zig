//! Arch-agnostic linear-scan register allocator (Poletto/Sarkar), see docs/ROADMAP.md §3.
//!
//! Input: a function as basic blocks of instructions, each listing the virtual registers it
//! uses and defines, plus call sites (caller-saved registers are clobbered) and optional
//! pre-colored (fixed) virtual registers. Output: one `Loc` per virtual register: a physical
//! register index or a stack spill slot.
//!
//! Steps:
//! 1. Liveness: backward dataflow over the CFG with bitsets (live-in/live-out per block).
//! 2. One live interval per vreg (no holes): [first def/live-in, last use/live-out], with
//!    positions 2*i for uses and 2*i+1 for defs, so a value dying at `i` and one born at `i`
//!    can share a register (`dst = src op ...`).
//! 3. Linear scan in order of interval start. Fixed vregs keep their register; other intervals
//!    avoid registers whose fixed intervals overlap them. Intervals that cross a call may only
//!    take callee-saved registers. When no register is free, the interval with the furthest
//!    end (the current one or an active one) is spilled. Spill slots are reused after expiry.
//! 4. `verify` checks that no two overlapping intervals share a register (run in tests and
//!    available to callers in Debug).
//!
//! Not yet: interval splitting, lifetime holes, edge move resolution, spill weights by loop
//! depth, two-address hints/coalescing, graph coloring.
const std = @import("std");

pub const VReg = u32;
pub const RegMask = u64;

pub const Inst = struct {
    uses: []const VReg = &.{},
    defs: []const VReg = &.{},
    /// Call site: registers outside `Config.callee_saved` are clobbered here.
    is_call: bool = false,
};

pub const Block = struct {
    insts: []const Inst,
    succs: []const u32 = &.{},
};

pub const Func = struct {
    blocks: []const Block,
    num_vregs: u32,
    /// Optional pre-colored vregs: `fixed[v] = reg` (length num_vregs or empty).
    fixed: []const ?u8 = &.{},
};

pub const Config = struct {
    /// Registers available for allocation (bit i = physical register i).
    allocatable: RegMask,
    /// Subset preserved across calls.
    callee_saved: RegMask = 0,
};

pub const Loc = union(enum) {
    none, // never defined or used
    reg: u8,
    stack: u32,
};

pub const Interval = struct { vreg: VReg, start: u32, end: u32 };

pub const Result = struct {
    locs: []Loc,
    intervals: []Interval,
    num_slots: u32,
    spills: u32,
    pub fn deinit(r: *Result, gpa: std.mem.Allocator) void {
        gpa.free(r.locs);
        gpa.free(r.intervals);
    }
};

pub const Error = error{ OutOfMemory, InvalidFunc, FixedConflict };

const none: u32 = std.math.maxInt(u32);
const SlotUse = struct { slot: u32, end: u32 };

/// Live-out sets per block (exposed for tests and the compiler).
pub fn liveness(gpa: std.mem.Allocator, f: Func) Error![]std.DynamicBitSetUnmanaged {
    const nb = f.blocks.len;
    const live_in = try gpa.alloc(std.DynamicBitSetUnmanaged, nb);
    defer {
        for (live_in) |*s| s.deinit(gpa);
        gpa.free(live_in);
    }
    const live_out = try gpa.alloc(std.DynamicBitSetUnmanaged, nb);
    for (0..nb) |i| {
        live_in[i] = try std.DynamicBitSetUnmanaged.initEmpty(gpa, f.num_vregs);
        live_out[i] = try std.DynamicBitSetUnmanaged.initEmpty(gpa, f.num_vregs);
    }
    var tmp = try std.DynamicBitSetUnmanaged.initEmpty(gpa, f.num_vregs);
    defer tmp.deinit(gpa);
    var changed = true;
    while (changed) {
        changed = false;
        var bi = nb;
        while (bi > 0) {
            bi -= 1;
            const b = f.blocks[bi];
            for (b.succs) |s| {
                if (s >= nb) return error.InvalidFunc;
                live_out[bi].setUnion(live_in[s]);
            }
            // in = uses ∪ (out − defs), walking instructions backwards
            tmp.unsetAll();
            tmp.setUnion(live_out[bi]);
            var ii = b.insts.len;
            while (ii > 0) {
                ii -= 1;
                for (b.insts[ii].defs) |d| tmp.unset(d);
                for (b.insts[ii].uses) |u| tmp.set(u);
            }
            if (!tmp.eql(live_in[bi])) {
                live_in[bi].unsetAll();
                live_in[bi].setUnion(tmp);
                changed = true;
            }
        }
    }
    return live_out;
}

fn extend(iv: []Interval, v: VReg, p: u32) void {
    if (iv[v].start == none or p < iv[v].start) iv[v].start = p;
    if (iv[v].end == none or p > iv[v].end) iv[v].end = p;
}

pub fn buildIntervals(gpa: std.mem.Allocator, f: Func) Error![]Interval {
    const live_out = try liveness(gpa, f);
    defer {
        for (live_out) |*s| s.deinit(gpa);
        gpa.free(live_out);
    }
    const iv = try gpa.alloc(Interval, f.num_vregs);
    for (iv, 0..) |*x, v| x.* = .{ .vreg = @intCast(v), .start = none, .end = none };
    var pos: u32 = 0; // instruction index across the whole function (block order = layout)
    for (f.blocks, 0..) |b, bi| {
        const first = pos;
        for (b.insts) |in| {
            for (in.uses) |u| {
                if (u >= f.num_vregs) return error.InvalidFunc;
                extend(iv, u, 2 * pos);
            }
            for (in.defs) |d| {
                if (d >= f.num_vregs) return error.InvalidFunc;
                extend(iv, d, 2 * pos + 1);
            }
            pos += 1;
        }
        const last_pos: u32 = if (pos == first) 2 * first else 2 * (pos - 1) + 1;
        var it = live_out[bi].iterator(.{});
        while (it.next()) |v| {
            extend(iv, @intCast(v), last_pos);
            // Live-out of a block => live through the whole block if also live-in; we keep
            // one interval without holes, so extending to the block start is conservative.
        }
    }
    // Live-in: a vreg live into block B is live from B's first position. With one interval
    // per vreg, extending to the start of every block where it is live-out covers loops.
    pos = 0;
    for (f.blocks, 0..) |b, bi| {
        const first = pos;
        pos += @intCast(b.insts.len);
        var it = live_out[bi].iterator(.{});
        while (it.next()) |v| extend(iv, @intCast(v), 2 * first);
    }
    return iv;
}

fn crossesCall(call_pos: []const u32, iv: Interval) bool {
    for (call_pos) |c| {
        if (iv.start < 2 * c and iv.end > 2 * c + 1) return true;
    }
    return false;
}

fn overlaps(a: Interval, b: Interval) bool {
    return a.start <= b.end and b.start <= a.end;
}

pub fn allocate(gpa: std.mem.Allocator, f: Func, cfg: Config) Error!Result {
    const iv = try buildIntervals(gpa, f);
    errdefer gpa.free(iv);
    const locs = try gpa.alloc(Loc, f.num_vregs);
    errdefer gpa.free(locs);
    @memset(locs, .none);

    var calls: std.ArrayList(u32) = .empty;
    defer calls.deinit(gpa);
    {
        var pos: u32 = 0;
        for (f.blocks) |b| for (b.insts) |in| {
            if (in.is_call) try calls.append(gpa, pos);
            pos += 1;
        };
    }

    // Fixed intervals per register.
    var fixed_iv: std.ArrayList(Interval) = .empty;
    defer fixed_iv.deinit(gpa);
    var fixed_reg: std.ArrayList(u8) = .empty;
    defer fixed_reg.deinit(gpa);
    for (f.fixed, 0..) |fx, v| if (fx) |r| {
        if (iv[v].start == none) continue;
        try fixed_iv.append(gpa, iv[v]);
        try fixed_reg.append(gpa, r);
        locs[v] = .{ .reg = r };
    };
    // Two fixed vregs on the same register must not overlap.
    for (fixed_iv.items, 0..) |a, i| for (fixed_iv.items[i + 1 ..], i + 1..) |b, j| {
        if (fixed_reg.items[i] == fixed_reg.items[j] and overlaps(a, b)) return error.FixedConflict;
    };

    // Unhandled intervals sorted by start.
    var order: std.ArrayList(Interval) = .empty;
    defer order.deinit(gpa);
    for (iv, 0..) |x, v| {
        if (x.start == none) continue;
        if (f.fixed.len > v and f.fixed[v] != null) continue;
        try order.append(gpa, x);
    }
    std.mem.sort(Interval, order.items, {}, struct {
        fn lt(_: void, a: Interval, b: Interval) bool {
            return a.start < b.start or (a.start == b.start and a.vreg < b.vreg);
        }
    }.lt);

    var active: std.ArrayList(Interval) = .empty; // sorted by end
    defer active.deinit(gpa);
    var free_slots: std.ArrayList(SlotUse) = .empty;
    defer free_slots.deinit(gpa);
    var slot_in_use: std.ArrayList(SlotUse) = .empty;
    defer slot_in_use.deinit(gpa);
    var num_slots: u32 = 0;
    var spills: u32 = 0;

    for (order.items) |cur| {
        // Expire old intervals.
        var k: usize = 0;
        while (k < active.items.len) {
            if (active.items[k].end < cur.start) {
                _ = active.orderedRemove(k);
            } else k += 1;
        }
        k = 0;
        while (k < slot_in_use.items.len) {
            if (slot_in_use.items[k].end < cur.start) {
                try free_slots.append(gpa, slot_in_use.swapRemove(k));
            } else k += 1;
        }

        var mask = cfg.allocatable;
        if (crossesCall(calls.items, cur)) mask &= cfg.callee_saved;
        for (active.items) |a| if (locs[a.vreg] == .reg) {
            mask &= ~(@as(RegMask, 1) << @intCast(locs[a.vreg].reg));
        };
        for (fixed_iv.items, fixed_reg.items) |fi, r| if (overlaps(fi, cur)) {
            mask &= ~(@as(RegMask, 1) << @intCast(r));
        };

        if (mask != 0) {
            locs[cur.vreg] = .{ .reg = @intCast(@ctz(mask)) };
            try insertByEnd(gpa, &active, cur);
            continue;
        }
        // Spill: the active interval with the furthest end if it ends after `cur` and its
        // register is usable by `cur`; otherwise spill `cur`.
        var victim: ?usize = null;
        var i = active.items.len;
        while (i > 0) {
            i -= 1;
            const a = active.items[i];
            if (a.end <= cur.end) break;
            const r = locs[a.vreg].reg;
            const bit = @as(RegMask, 1) << @intCast(r);
            var ok = cfg.allocatable & bit != 0;
            if (crossesCall(calls.items, cur) and cfg.callee_saved & bit == 0) ok = false;
            for (fixed_iv.items, fixed_reg.items) |fi, fr| if (fr == r and overlaps(fi, cur)) {
                ok = false;
            };
            if (ok) {
                victim = i;
                break;
            }
        }
        const spilled: Interval = if (victim) |vi| blk: {
            const a = active.orderedRemove(vi);
            locs[cur.vreg] = locs[a.vreg];
            try insertByEnd(gpa, &active, cur);
            break :blk a;
        } else cur;
        // Reuse a slot only if its previous owner ended before the spilled interval began
        // (a spilled active interval started before `cur`).
        const slot: u32 = blk: {
            for (free_slots.items, 0..) |s, si| if (s.end < spilled.start) {
                break :blk free_slots.swapRemove(si).slot;
            };
            num_slots += 1;
            break :blk num_slots - 1;
        };
        locs[spilled.vreg] = .{ .stack = slot };
        try slot_in_use.append(gpa, .{ .slot = slot, .end = spilled.end });
        spills += 1;
    }
    return .{ .locs = locs, .intervals = iv, .num_slots = num_slots, .spills = spills };
}

fn insertByEnd(gpa: std.mem.Allocator, list: *std.ArrayList(Interval), x: Interval) !void {
    var i: usize = 0;
    while (i < list.items.len and list.items[i].end <= x.end) i += 1;
    try list.insert(gpa, i, x);
}

/// Interference checker: overlapping intervals never share a register or a spill slot, and
/// intervals crossing a call only use callee-saved registers.
pub fn verify(f: Func, cfg: Config, r: Result) bool {
    var pos: u32 = 0;
    for (f.blocks) |b| for (b.insts) |in| {
        if (in.is_call) for (r.intervals) |x| {
            if (x.start == none) continue;
            if (x.start < 2 * pos and x.end > 2 * pos + 1) switch (r.locs[x.vreg]) {
                .reg => |reg| if (cfg.callee_saved & (@as(RegMask, 1) << @intCast(reg)) == 0) {
                    const is_fixed = f.fixed.len > x.vreg and f.fixed[x.vreg] != null;
                    if (!is_fixed) return false;
                },
                else => {},
            };
        };
        pos += 1;
    };
    for (r.intervals, 0..) |a, i| {
        if (a.start == none) continue;
        for (r.intervals[i + 1 ..]) |b| {
            if (b.start == none or !overlaps(a, b)) continue;
            const la = r.locs[a.vreg];
            const lb = r.locs[b.vreg];
            if (la == .reg and lb == .reg and la.reg == lb.reg) return false;
            if (la == .stack and lb == .stack and la.stack == lb.stack) return false;
        }
    }
    return true;
}

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;

test "regalloc: liveness across a loop" {
    // b0: v0 = ...; v1 = ...        -> b1
    // b1: use v0, v1; v1 = v1 + v0  -> b1, b2
    // b2: use v1 (ret)
    const blocks = [_]Block{
        .{ .insts = &.{ .{ .defs = &.{0} }, .{ .defs = &.{1} } }, .succs = &.{1} },
        .{ .insts = &.{.{ .uses = &.{ 1, 0 }, .defs = &.{1} }}, .succs = &.{ 1, 2 } },
        .{ .insts = &.{.{ .uses = &.{1} }} },
    };
    const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
    const lo = try liveness(t.allocator, f);
    defer {
        for (lo) |*s| s.deinit(t.allocator);
        t.allocator.free(lo);
    }
    try t.expect(lo[0].isSet(0) and lo[0].isSet(1));
    try t.expect(lo[1].isSet(0) and lo[1].isSet(1)); // v0 still needed by the next iteration
    try t.expect(!lo[2].isSet(0) and !lo[2].isSet(1));
    var r = try allocate(t.allocator, f, .{ .allocatable = 0b11 });
    defer r.deinit(t.allocator);
    try t.expect(verify(f, .{ .allocatable = 0b11 }, r));
    try t.expectEqual(@as(u32, 0), r.spills);
    try t.expect(r.locs[0].reg != r.locs[1].reg);
}

test "regalloc: dying source and new def share a register" {
    const blocks = [_]Block{.{ .insts = &.{ .{ .defs = &.{0} }, .{ .uses = &.{0}, .defs = &.{1} }, .{ .uses = &.{1} } } }};
    const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
    var r = try allocate(t.allocator, f, .{ .allocatable = 0b1 });
    defer r.deinit(t.allocator);
    try t.expectEqual(@as(u32, 0), r.spills);
    try t.expectEqual(Loc{ .reg = 0 }, r.locs[1]);
}

test "regalloc: spill stress, 50 live values in 3 registers" {
    const n = 50;
    var insts: [n + 1]Inst = undefined;
    var ids: [n]VReg = undefined;
    for (0..n) |i| {
        ids[i] = @intCast(i);
        insts[i] = .{ .defs = ids[i .. i + 1] };
    }
    insts[n] = .{ .uses = &ids }; // all live at once
    const blocks = [_]Block{.{ .insts = &insts }};
    const f: Func = .{ .blocks = &blocks, .num_vregs = n };
    const cfg: Config = .{ .allocatable = 0b111 };
    var r = try allocate(t.allocator, f, cfg);
    defer r.deinit(t.allocator);
    try t.expect(verify(f, cfg, r));
    try t.expectEqual(@as(u32, n - 3), r.spills);
    try t.expectEqual(@as(u32, n - 3), r.num_slots);
}

test "regalloc: values live across a call get callee-saved registers" {
    // v0 defined, call, use v0; v1 short-lived around nothing.
    const blocks = [_]Block{.{ .insts = &.{
        .{ .defs = &.{0} },
        .{ .defs = &.{1} },
        .{ .uses = &.{1}, .is_call = true },
        .{ .uses = &.{0} },
    } }};
    const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
    const cfg: Config = .{ .allocatable = 0b1111, .callee_saved = 0b1000 };
    var r = try allocate(t.allocator, f, cfg);
    defer r.deinit(t.allocator);
    try t.expect(verify(f, cfg, r));
    try t.expectEqual(Loc{ .reg = 3 }, r.locs[0]);
    try t.expectEqual(Loc{ .reg = 0 }, r.locs[1]);
    // With no callee-saved register the crossing value is spilled.
    var r2 = try allocate(t.allocator, f, .{ .allocatable = 0b1111 });
    defer r2.deinit(t.allocator);
    try t.expect(r2.locs[0] == .stack);
}

test "regalloc: fixed registers are respected and avoided by others" {
    // v0 fixed in r1 (e.g. a shift count), v1/v2 free; all overlap.
    const blocks = [_]Block{.{ .insts = &.{
        .{ .defs = &.{1} },
        .{ .defs = &.{0} },
        .{ .defs = &.{2} },
        .{ .uses = &.{ 0, 1, 2 } },
    } }};
    const fixed = [_]?u8{ 1, null, null };
    const f: Func = .{ .blocks = &blocks, .num_vregs = 3, .fixed = &fixed };
    const cfg: Config = .{ .allocatable = 0b111 };
    var r = try allocate(t.allocator, f, cfg);
    defer r.deinit(t.allocator);
    try t.expect(verify(f, cfg, r));
    try t.expectEqual(Loc{ .reg = 1 }, r.locs[0]);
    try t.expect(r.locs[1].reg != 1 and r.locs[2].reg != 1);
    const bad = [_]?u8{ 1, 1, null };
    try t.expectError(error.FixedConflict, allocate(t.allocator, .{ .blocks = &blocks, .num_vregs = 3, .fixed = &bad }, cfg));
}

test "regalloc: random programs always verify" {
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    for (0..300) |_| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const A = arena.allocator();
        const nv: u32 = 1 + rnd.uintLessThan(u32, 40);
        const nb: usize = 1 + rnd.uintLessThan(usize, 6);
        const blocks = try A.alloc(Block, nb);
        for (blocks, 0..) |*b, bi| {
            const ni = 1 + rnd.uintLessThan(usize, 12);
            const insts = try A.alloc(Inst, ni);
            for (insts) |*in| {
                const u = try A.alloc(VReg, rnd.uintLessThan(usize, 3));
                for (u) |*x| x.* = rnd.uintLessThan(u32, nv);
                const d = try A.alloc(VReg, rnd.uintLessThan(usize, 2));
                for (d) |*x| x.* = rnd.uintLessThan(u32, nv);
                in.* = .{ .uses = u, .defs = d, .is_call = rnd.uintLessThan(u8, 8) == 0 };
            }
            const ns = rnd.uintLessThan(usize, 3);
            const s = try A.alloc(u32, ns);
            for (s) |*x| x.* = rnd.uintLessThan(u32, @intCast(nb));
            b.* = .{ .insts = insts, .succs = s };
            _ = bi;
        }
        const f: Func = .{ .blocks = blocks, .num_vregs = nv };
        const cfg: Config = .{ .allocatable = 0b11111, .callee_saved = 0b11000 };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
    }
}
