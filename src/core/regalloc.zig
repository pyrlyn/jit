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
//! 5. Splitting at calls (`Config.split_at_calls`, first version of live-range splitting): a
//!    call-crossing interval that finds no free callee-saved register may take a caller-saved
//!    one; its range is split around each call it crosses: the value is stored to
//!    `Result.save_slot[v]` before the call and reloaded after it (by the code generator,
//!    using `crossesCallAt`). Callee-saved registers are still preferred.
//!
//! 6. Graph coloring (`Config.algo = .graph`, Chaitin-Briggs with optimistic coloring, no
//!    coalescing): the interference graph is built from the same intervals; nodes with fewer
//!    neighbours than usable registers are simplified first, otherwise the node with the lowest
//!    spill weight ((uses + defs) / length) is pushed optimistically; select assigns the lowest
//!    free register (callee-saved first for call-crossing values) or spills. Spilled values
//!    share stack slots by colouring the spilled subgraph. Same `Result` and `verify`.
//!
//! 7. Live-range splitting of spilled values (`Config.split`): a spilled value keeps its stack
//!    slot as its home (so block boundaries never need edge moves), but inside each block where
//!    it occurs at least twice it gets a register segment from its first to its last occurrence
//!    when one is free over that span (callee-saved if the span crosses a call). The code
//!    generator loads the register at the segment start when the first occurrence is a use
//!    (`Seg.load`), reads it from the register inside the segment, and writes defs through to
//!    both the register and the slot (`Result.regAt`, `Result.loadsAt`).
//!
//! 8. Coalescing (`.graph` only, `Inst.is_copy`): copy-related, non-interfering nodes are merged
//!    before simplify when Briggs' conservative test holds (the merged node has fewer than K
//!    neighbours of significant degree), so both ends of the copy get the same register and the
//!    move disappears (`Result.coalesced`).
//!
//! Not yet: lifetime holes, splitting of register-allocated intervals, spill weights by loop
//! depth.
const std = @import("std");

pub const VReg = u32;
pub const RegMask = u64;

pub const Inst = struct {
    uses: []const VReg = &.{},
    defs: []const VReg = &.{},
    /// Call site: registers outside `Config.callee_saved` are clobbered here.
    is_call: bool = false,
    /// `defs[0] = uses[0]` (a move; coalescing candidate for `.graph`).
    is_copy: bool = false,
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

/// Compact compiler input: operand lists share a flat array. Public slice-based `Func`
/// remains available for callers that build allocation inputs themselves.
pub const CompactInst = struct {
    offset: u32,
    nuses: u8,
    ndefs: u8,
    is_call: bool = false,
    is_copy: bool = false,
};
pub const CompactBlock = struct {
    insts: []const CompactInst,
    succs: []const u32 = &.{},
};
pub const CompactFunc = struct {
    blocks: []const CompactBlock,
    operands: []const VReg,
    num_vregs: u32,
    fixed: []const ?u8 = &.{},
};

inline fn uses(f: anytype, in: anytype) []const VReg {
    if (@TypeOf(f) == Func) return in.uses;
    return f.operands[in.offset..][0..in.nuses];
}
inline fn defs(f: anytype, in: anytype) []const VReg {
    if (@TypeOf(f) == Func) return in.defs;
    return f.operands[@as(usize, in.offset) + in.nuses ..][0..in.ndefs];
}

inline fn isCall(f: anytype, b: anytype, i: usize) bool {
    if (@hasDecl(@TypeOf(f), "isCall")) return f.isCall(b, i);
    return b.insts[i].is_call;
}

/// Compiler-only linear input: collectBlock visits operands at their original
/// positions, and isCall reads the tag without materializing operand lists.
pub fn allocateLinearSource(gpa: std.mem.Allocator, f: anytype, cfg: Config) Error!Result {
    std.debug.assert(cfg.algo == .linear and !cfg.split);
    return allocateLinearInput(gpa, f, cfg);
}
pub fn verifySource(f: anytype, cfg: Config, r: Result) bool {
    return verifyInput(f, cfg, r);
}

pub const Config = struct {
    /// Registers available for allocation (bit i = physical register i).
    allocatable: RegMask,
    /// Subset preserved across calls.
    callee_saved: RegMask = 0,
    /// Allow caller-saved registers for call-crossing intervals, saved/restored around calls.
    split_at_calls: bool = false,
    /// Allocation algorithm.
    algo: Algo = .linear,
    /// Give spilled values per-block register segments (step 7).
    split: bool = false,
    /// Coalesce copies in `.graph` (step 8).
    coalesce: bool = true,
};

/// Register segment of a spilled vreg: instructions `first..last` (inclusive, global index).
pub const Seg = struct { vreg: VReg, first: u32, last: u32, reg: u8, load: bool };

pub const Algo = enum { linear, graph };

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
    /// Per vreg: spill slot used to save its caller-saved register around calls (split range).
    save_slot: []?u32,
    /// Register segments of spilled vregs, sorted by (vreg, first); `seg_start[v]..seg_start[v+1]`.
    segs: []const Seg = &.{},
    seg_start: []const u32 = &.{},
    /// Indices into `segs` of the segments with `load`, sorted by (first, vreg).
    load_order: []const u32 = &.{},
    /// Copies removed by coalescing (both ends share a register).
    coalesced: u32 = 0,
    /// Vregs live into the entry block: read on some path before any def. The compiler sets
    /// them to 0 after the parameter moves (as the IR interpreter does), ascending.
    entry_undef: []const VReg = &.{},
    pub fn deinit(r: *Result, gpa: std.mem.Allocator) void {
        if (r.entry_undef.len != 0) gpa.free(r.entry_undef);
        gpa.free(r.locs);
        gpa.free(r.intervals);
        gpa.free(r.save_slot);
        if (r.seg_start.len != 0) {
            gpa.free(r.segs);
            gpa.free(r.seg_start);
            gpa.free(r.load_order);
        }
    }
    fn segsOf(r: Result, v: VReg) []const Seg {
        if (r.seg_start.len == 0) return &.{};
        return r.segs[r.seg_start[v]..r.seg_start[v + 1]];
    }
    /// Register holding spilled `v` at instruction `pos` (inside one of its segments), if any.
    pub fn regAt(r: Result, v: VReg, pos: u32) ?u8 {
        for (r.segsOf(v)) |sg| if (sg.first <= pos and pos <= sg.last) return sg.reg;
        return null;
    }
    /// Calls `f(ctx, vreg, reg)` for each segment that must be loaded from its slot before `pos`.
    pub fn loadsAt(r: Result, pos: u32, ctx: anytype, comptime f: anytype) !void {
        const Ctx = struct { segs: []const Seg, pos: u32 };
        var i = std.sort.partitionPoint(u32, r.load_order, Ctx{ .segs = r.segs, .pos = pos }, struct {
            fn before(c: Ctx, k: u32) bool {
                return c.segs[k].first < c.pos;
            }
        }.before);
        while (i < r.load_order.len and r.segs[r.load_order[i]].first == pos) : (i += 1) {
            const sg = r.segs[r.load_order[i]];
            try f(ctx, sg.vreg, sg.reg);
        }
    }
    /// All registers used by `locs` or segments (for callee-saved register saving).
    pub fn usedRegs(r: Result) RegMask {
        var m: RegMask = 0;
        for (r.locs) |l| if (l == .reg) {
            m |= @as(RegMask, 1) << @intCast(l.reg);
        };
        for (r.segs) |sg| m |= @as(RegMask, 1) << @intCast(sg.reg);
        return m;
    }
    /// True if `v` must be saved/restored around the call at instruction position `pos`.
    pub fn savedAcross(r: Result, v: VReg, pos: u32) bool {
        return r.save_slot[v] != null and crossesCallAt(r.intervals[v], pos);
    }
};

pub const Error = error{ OutOfMemory, InvalidFunc, FixedConflict };

const none: u32 = std.math.maxInt(u32);
const SlotUse = struct { slot: u32, end: u32 };

/// Live-out sets per block (exposed for tests and the compiler).
pub fn liveness(gpa: std.mem.Allocator, f: Func) Error![]std.DynamicBitSetUnmanaged {
    const nb = f.blocks.len;
    const nw = liveWords(f);
    const out = try liveOut(gpa, f);
    defer gpa.free(out);
    const live_out = try gpa.alloc(std.DynamicBitSetUnmanaged, nb);
    var made: usize = 0;
    errdefer {
        for (live_out[0..made]) |*s| s.deinit(gpa);
        gpa.free(live_out);
    }
    for (live_out, 0..) |*s, bi| {
        s.* = try std.DynamicBitSetUnmanaged.initEmpty(gpa, f.num_vregs);
        made += 1;
        @memcpy(s.masks[0..nw], out[bi * nw ..][0..nw]);
    }
    return live_out;
}

const LiveWord = usize;
fn liveWords(f: anytype) usize {
    return (@as(usize, f.num_vregs) + @bitSizeOf(LiveWord) - 1) / @bitSizeOf(LiveWord);
}

/// Live-out sets as one flat word array: block `bi` is `out[bi * liveWords(f) ..][0..liveWords(f)]`,
/// followed by the entry block's live-in set (`liveWords(f)` more words).
fn liveOut(gpa: std.mem.Allocator, f: anytype) Error![]LiveWord {
    return liveOutIntervals(gpa, f, false, &.{});
}

// Build local interval endpoints while collecting gen/kill: both need the same
// forward uses/defs walk. `liveness` callers can omit intervals.
fn liveOutIntervals(gpa: std.mem.Allocator, f: anytype, comptime with_intervals: bool, iv: []Interval) Error![]LiveWord {
    const nb = f.blocks.len;
    for (f.blocks) |b| for (b.succs) |succ| if (succ >= nb) return error.InvalidFunc;
    // Per block: gen (upward-exposed uses), kill (defs) and live-in, as flat word arrays, so
    // the fixpoint iterates over words only: in = gen ∪ (out − kill).
    const W = LiveWord;
    const bits = @bitSizeOf(W);
    const nw = liveWords(f);
    // gen | kill | in, then `out` (returned) in its own allocation.
    const flat = try gpa.alloc(W, 3 * nb * nw);
    defer gpa.free(flat);
    @memset(flat, 0);
    const gen = flat[0 .. nb * nw];
    const kill = flat[nb * nw .. 2 * nb * nw];
    const in_ = flat[2 * nb * nw .. 3 * nb * nw];
    const out = try gpa.alloc(W, (nb + 1) * nw);
    errdefer gpa.free(out);
    @memset(out, 0);
    const Collector = struct {
        g: []W,
        k: []W,
        iv: []Interval,
        n: u32,
        first: u32,
        pos: u32,
        pub inline fn use(c: *@This(), u: VReg) Error!void {
            if (u >= c.n) return error.InvalidFunc;
            const bit = @as(W, 1) << @intCast(u % bits);
            const exposed = c.k[u / bits] & bit == 0;
            c.g[u / bits] |= bit & ~c.k[u / bits];
            if (with_intervals) {
                c.iv[u].start = @min(c.iv[u].start, 2 * (if (exposed) c.first else c.pos));
                c.iv[u].end = 2 * c.pos;
            }
        }
        pub inline fn def(c: *@This(), d: VReg) Error!void {
            if (d >= c.n) return error.InvalidFunc;
            c.k[d / bits] |= @as(W, 1) << @intCast(d % bits);
            if (with_intervals) {
                c.iv[d].start = @min(c.iv[d].start, 2 * c.pos + 1);
                c.iv[d].end = 2 * c.pos + 1;
            }
        }
        pub inline fn next(c: *@This()) void {
            c.pos += 1;
        }
    };
    var pos: u32 = 0;
    for (f.blocks, 0..) |b, bi| {
        var c: Collector = .{ .g = gen[bi * nw ..][0..nw], .k = kill[bi * nw ..][0..nw], .iv = iv, .n = f.num_vregs, .first = pos, .pos = pos };
        if (@hasDecl(@TypeOf(f), "collectBlock")) {
            try f.collectBlock(b, &c);
        } else {
            for (b.insts) |in| {
                for (uses(f, in)) |u| try c.use(u);
                for (defs(f, in)) |d| try c.def(d);
                c.next();
            }
        }
        pos = c.pos;
    }
    var changed = true;
    while (changed) {
        changed = false;
        var bi = nb;
        while (bi > 0) {
            bi -= 1;
            const o = out[bi * nw ..][0..nw];
            for (f.blocks[bi].succs) |s| {
                for (o, in_[s * nw ..][0..nw]) |*ow, iw| ow.* |= iw;
            }
            const g = gen[bi * nw ..][0..nw];
            const k = kill[bi * nw ..][0..nw];
            for (in_[bi * nw ..][0..nw], g, k, o) |*iw, gw, kw, ow| {
                const nv = gw | (ow & ~kw);
                if (nv != iw.*) {
                    iw.* = nv;
                    changed = true;
                }
            }
        }
    }
    if (nb != 0) @memcpy(out[nb * nw ..], in_[0..nw]);
    return out;
}

pub fn buildIntervals(gpa: std.mem.Allocator, f: Func) Error![]Interval {
    return buildIntervalsUndef(gpa, f, null);
}

/// `buildIntervals`, plus the entry block's live-in vregs in `undef` (owned by `gpa` when
/// non-empty; see `Result.entry_undef`).
fn buildIntervalsUndef(gpa: std.mem.Allocator, f: anytype, undef: ?*[]VReg) Error![]Interval {
    const iv = try gpa.alloc(Interval, f.num_vregs);
    errdefer gpa.free(iv);
    for (iv, 0..) |*x, v| x.* = .{ .vreg = @intCast(v), .start = none, .end = none };
    const live_out = try liveOutIntervals(gpa, f, true, iv);
    defer gpa.free(live_out);
    const nw = liveWords(f);
    var pos: u32 = 0;
    for (f.blocks, 0..) |b, bi| {
        const first = pos;
        pos += @intCast(b.insts.len);
        const last_pos: u32 = if (pos == first) 2 * first else 2 * (pos - 1) + 1;
        // Live-out of a block: live to its end and, with one interval per vreg and no
        // holes, conservatively from its start too (covers loops). Both in one pass.
        for (live_out[bi * nw ..][0..nw], 0..) |w0, wi| {
            // Interior blocks of a consecutive live-out run cannot change either
            // endpoint. Visit only its first/last block, preserving lifetime holes.
            const prev = if (bi == 0) 0 else live_out[(bi - 1) * nw + wi];
            const next = if (bi + 1 == f.blocks.len) 0 else live_out[(bi + 1) * nw + wi];
            var w = w0 & ~(prev & next);
            while (w != 0) : (w &= w - 1) {
                const x = &iv[wi * @bitSizeOf(LiveWord) + @ctz(w)];
                if (x.start == none or 2 * first < x.start) x.start = 2 * first;
                if (x.end == none or last_pos > x.end) x.end = last_pos;
            }
        }
    }
    if (undef) |u| {
        const entry_in = live_out[f.blocks.len * nw ..][0..nw];
        var cnt: usize = 0;
        for (entry_in) |w| cnt += @popCount(w);
        u.* = &.{};
        if (cnt != 0) {
            const list = try gpa.alloc(VReg, cnt);
            var k: usize = 0;
            for (entry_in, 0..) |w0, wi| {
                var w = w0;
                while (w != 0) : (w &= w - 1) {
                    list[k] = @intCast(wi * @bitSizeOf(LiveWord) + @ctz(w));
                    k += 1;
                }
            }
            u.* = list;
        }
    }
    return iv;
}

/// Interval is live before and after the instruction at position `pos`.
pub fn crossesCallAt(iv: Interval, pos: u32) bool {
    return iv.start != none and iv.start < 2 * pos and iv.end > 2 * pos + 1;
}

/// `call_pos` is sorted (ascending instruction positions).
fn crossesCall(call_pos: []const u32, iv: Interval) bool {
    // First call with 2c > start; it is crossed iff 2c + 1 < end (later calls end later).
    const i = std.sort.partitionPoint(u32, call_pos, iv.start, struct {
        fn f(start: u32, c: u32) bool {
            return 2 * c <= start;
        }
    }.f);
    return i < call_pos.len and iv.end > 2 * call_pos[i] + 1;
}

fn overlaps(a: Interval, b: Interval) bool {
    return a.start <= b.end and b.start <= a.end;
}

pub fn allocate(gpa: std.mem.Allocator, f: Func, cfg: Config) Error!Result {
    return allocateInput(gpa, f, cfg);
}

fn allocateInput(gpa: std.mem.Allocator, f: anytype, cfg: Config) Error!Result {
    var r = switch (cfg.algo) {
        .linear => try allocateLinearInput(gpa, f, cfg),
        .graph => try allocateGraphInput(gpa, f, cfg),
    };
    if (cfg.split) {
        errdefer r.deinit(gpa);
        try splitSpilled(gpa, f, cfg, &r);
    }
    return r;
}

/// Step 7: per-block register segments for spilled vregs (see header).
fn splitSpilled(gpa: std.mem.Allocator, f: anytype, cfg: Config, r: *Result) Error!void {
    const n = f.num_vregs;
    var calls = try callPositions(gpa, f);
    defer calls.deinit(gpa);
    var segs: std.ArrayList(Seg) = .empty;
    defer segs.deinit(gpa);
    // Per block, per spilled vreg: first/last occurrence and count.
    const first = try gpa.alloc(u32, n);
    defer gpa.free(first);
    const last = try gpa.alloc(u32, n);
    defer gpa.free(last);
    const cnt = try gpa.alloc(u32, n);
    defer gpa.free(cnt);
    const first_is_use = try gpa.alloc(bool, n);
    defer gpa.free(first_is_use);
    @memset(cnt, 0);
    // Spilled vregs touched in the current block (in first-touch order; sorted before use so
    // segments are created in vreg order as before).
    var touched: std.ArrayList(VReg) = .empty;
    defer touched.deinit(gpa);
    // Register-allocated intervals overlapping the current block (segments never leave it).
    var reg_ivs: std.ArrayList(u32) = .empty;
    defer reg_ivs.deinit(gpa);
    var reg_all: std.ArrayList(u32) = .empty;
    defer reg_all.deinit(gpa);
    for (r.intervals, 0..) |w, wi| if (w.start != none and r.locs[wi] == .reg) try reg_all.append(gpa, @intCast(wi));
    var base: u32 = 0;
    var ci: usize = 0; // first call at or after `base`
    for (f.blocks) |b| {
        const len: u32 = @intCast(b.insts.len);
        defer base += len;
        if (len == 0) continue;
        touched.clearRetainingCapacity();
        for (b.insts, 0..) |in, k| {
            const p = base + @as(u32, @intCast(k));
            for (uses(f, in)) |u| if (r.locs[u] == .stack) {
                if (cnt[u] == 0) try touched.append(gpa, u);
                note(first, last, cnt, first_is_use, u, p, true);
            };
            for (defs(f, in)) |d| if (r.locs[d] == .stack) {
                if (cnt[d] == 0) try touched.append(gpa, d);
                note(first, last, cnt, first_is_use, d, p, false);
            };
        }
        while (ci < calls.items.len and calls.items[ci] < base) ci += 1;
        var ce = ci;
        while (ce < calls.items.len and calls.items[ce] < base + len) ce += 1;
        const block_calls = calls.items[ci..ce];
        const block_span: Interval = .{ .vreg = 0, .start = 2 * base, .end = 2 * (base + len - 1) + 1 };
        reg_ivs.clearRetainingCapacity();
        for (reg_all.items) |wi| if (overlaps(r.intervals[wi], block_span)) try reg_ivs.append(gpa, wi);
        const block_segs = segs.items.len;
        std.mem.sort(VReg, touched.items, {}, std.sort.asc(VReg));
        for (touched.items) |v| {
            defer cnt[v] = 0;
            if (cnt[v] < 2) continue;
            if (f.fixed.len > v and f.fixed[v] != null) continue;
            const span: Interval = .{ .vreg = v, .start = 2 * first[v], .end = 2 * last[v] + 1 };
            var mask = cfg.allocatable;
            // A segment loaded for a call argument (first == c, a use) is live in its register
            // before the call and read again after it: crossing too.
            for (block_calls) |c| if ((first[v] < c or (first[v] == c and first_is_use[v])) and c < last[v]) {
                mask &= cfg.callee_saved;
                break;
            };
            for (reg_ivs.items) |wi| if (overlaps(r.intervals[wi], span)) {
                mask &= ~(@as(RegMask, 1) << @intCast(r.locs[wi].reg));
            };
            for (segs.items[block_segs..]) |sg| if (sg.first <= last[v] and first[v] <= sg.last) {
                mask &= ~(@as(RegMask, 1) << @intCast(sg.reg));
            };
            if (mask == 0) continue;
            try segs.append(gpa, .{ .vreg = v, .first = first[v], .last = last[v], .reg = @intCast(@ctz(mask)), .load = first_is_use[v] });
        }
    }
    std.mem.sort(Seg, segs.items, {}, struct {
        fn lt(_: void, a: Seg, b: Seg) bool {
            return a.vreg < b.vreg or (a.vreg == b.vreg and a.first < b.first);
        }
    }.lt);
    const start = try gpa.alloc(u32, n + 1);
    errdefer gpa.free(start);
    var k: u32 = 0;
    for (0..n + 1) |v| {
        while (k < segs.items.len and segs.items[k].vreg < v) k += 1;
        start[v] = k;
    }
    // Segments that need a load, by (first, vreg): `loadsAt` finds them by binary search.
    var nload: usize = 0;
    for (segs.items) |sg| nload += @intFromBool(sg.load);
    const order = try gpa.alloc(u32, nload);
    errdefer gpa.free(order);
    nload = 0;
    for (segs.items, 0..) |sg, i| if (sg.load) {
        order[nload] = @intCast(i);
        nload += 1;
    };
    std.mem.sort(u32, order, segs.items, struct {
        fn lt(s: []const Seg, a: u32, b: u32) bool {
            return s[a].first < s[b].first or (s[a].first == s[b].first and s[a].vreg < s[b].vreg);
        }
    }.lt);
    r.segs = try segs.toOwnedSlice(gpa);
    r.seg_start = start;
    r.load_order = order;
}

fn note(first: []u32, last: []u32, cnt: []u32, first_is_use: []bool, v: VReg, p: u32, use: bool) void {
    if (cnt[v] == 0) {
        first[v] = p;
        first_is_use[v] = use;
    } else if (first[v] == p and use) first_is_use[v] = true;
    last[v] = p;
    cnt[v] += 1;
}

fn callPositions(gpa: std.mem.Allocator, f: anytype) Error!std.ArrayList(u32) {
    var calls: std.ArrayList(u32) = .empty;
    errdefer calls.deinit(gpa);
    var pos: u32 = 0;
    for (f.blocks) |b| for (0..b.insts.len) |ii| {
        if (isCall(f, b, ii)) try calls.append(gpa, pos);
        pos += 1;
    };
    return calls;
}

/// Caller-saved registers holding call-crossing values get a save slot (split ranges).
fn saveSlots(gpa: std.mem.Allocator, f: anytype, cfg: Config, iv: []const Interval, locs: []const Loc, calls: []const u32, num_slots: *u32) Error![]?u32 {
    const save = try gpa.alloc(?u32, f.num_vregs);
    @memset(save, null);
    for (iv, 0..) |x, v| {
        if (locs[v] != .reg or !crossesCall(calls, x)) continue;
        if (cfg.callee_saved & (@as(RegMask, 1) << @intCast(locs[v].reg)) != 0) continue;
        if (f.fixed.len > v and f.fixed[v] != null) continue;
        save[v] = num_slots.*;
        num_slots.* += 1;
    }
    return save;
}

/// Chaitin-Briggs graph colouring over the interval interference graph (see header, step 6).
pub fn allocateGraph(gpa: std.mem.Allocator, f: Func, cfg: Config) Error!Result {
    return allocateGraphInput(gpa, f, cfg);
}

fn allocateGraphInput(gpa: std.mem.Allocator, f: anytype, cfg: Config) Error!Result {
    const n = f.num_vregs;
    var undef: []VReg = &.{};
    const iv = try buildIntervalsUndef(gpa, f, &undef);
    errdefer gpa.free(iv);
    errdefer if (undef.len != 0) gpa.free(undef);
    const locs = try gpa.alloc(Loc, n);
    errdefer gpa.free(locs);
    @memset(locs, .none);
    var calls = try callPositions(gpa, f);
    defer calls.deinit(gpa);

    const isFixed = struct {
        fn f_(fx: []const ?u8, v: usize) bool {
            return fx.len > v and fx[v] != null;
        }
    }.f_;
    for (f.fixed, 0..) |fx, v| if (fx) |r| {
        if (iv[v].start == none) continue;
        for (f.fixed[v + 1 ..], v + 1..) |fy, w| if (fy) |r2| {
            if (r == r2 and iv[w].start != none and overlaps(iv[v], iv[w])) return error.FixedConflict;
        };
        locs[v] = .{ .reg = r };
    };

    // Spill weights: occurrences / interval length.
    const occ = try gpa.alloc(u32, n);
    defer gpa.free(occ);
    @memset(occ, 0);
    for (f.blocks) |b| for (b.insts) |in| {
        for (uses(f, in)) |u| occ[u] += 1;
        for (defs(f, in)) |d| occ[d] += 1;
    };

    // Nodes, allowed masks, adjacency (bitsets over vregs).
    const allowed = try gpa.alloc(RegMask, n);
    defer gpa.free(allowed);
    const crosses = try gpa.alloc(bool, n);
    defer gpa.free(crosses);
    const adj = try gpa.alloc(std.DynamicBitSetUnmanaged, n);
    var made: usize = 0;
    defer {
        for (adj[0..made]) |*a| a.deinit(gpa);
        gpa.free(adj);
    }
    for (adj) |*a| {
        a.* = try std.DynamicBitSetUnmanaged.initEmpty(gpa, n);
        made += 1;
    }
    for (0..n) |v| {
        crosses[v] = iv[v].start != none and crossesCall(calls.items, iv[v]);
        var m = cfg.allocatable;
        if (crosses[v] and !cfg.split_at_calls) m &= cfg.callee_saved;
        allowed[v] = m;
    }
    for (0..n) |v| {
        if (iv[v].start == none) continue;
        for (v + 1..n) |w| {
            if (iv[w].start == none or !overlaps(iv[v], iv[w])) continue;
            const fv = isFixed(f.fixed, v);
            const fw = isFixed(f.fixed, w);
            if (fv and fw) continue;
            if (fv) {
                allowed[w] &= ~(@as(RegMask, 1) << @intCast(f.fixed[v].?));
            } else if (fw) {
                allowed[v] &= ~(@as(RegMask, 1) << @intCast(f.fixed[w].?));
            } else {
                adj[v].set(w);
                adj[w].set(v);
            }
        }
    }

    // Coalesce (Briggs): merge copy-related, non-interfering nodes.
    const alias = try gpa.alloc(u32, n);
    defer gpa.free(alias);
    for (alias, 0..) |*a, v| a.* = @intCast(v);
    var coalesced: u32 = 0;
    var both = try std.DynamicBitSetUnmanaged.initEmpty(gpa, n);
    defer both.deinit(gpa);
    if (cfg.coalesce) for (f.blocks) |b| for (b.insts) |in| {
        if (!in.is_copy or uses(f, in).len != 1 or defs(f, in).len != 1) continue;
        const x0 = root(alias, uses(f, in)[0]);
        const y0 = root(alias, defs(f, in)[0]);
        if (x0 == y0 or isFixed(f.fixed, x0) or isFixed(f.fixed, y0)) continue;
        if (iv[x0].start == none or iv[y0].start == none or adj[x0].isSet(y0)) continue;
        const m = allowed[x0] & allowed[y0];
        const kk: u32 = @popCount(m);
        if (kk == 0) continue;
        both.unsetAll();
        both.setUnion(adj[x0]);
        both.setUnion(adj[y0]);
        var sig: u32 = 0;
        var it = both.iterator(.{});
        while (it.next()) |w| if (adj[w].count() >= kk) {
            sig += 1;
        };
        if (sig >= kk) continue;
        // Merge y0 into x0.
        var it2 = adj[y0].iterator(.{});
        while (it2.next()) |w| {
            adj[w].unset(y0);
            adj[w].set(x0);
        }
        adj[x0].setUnion(adj[y0]);
        adj[y0].unsetAll();
        allowed[x0] = m;
        crosses[x0] = crosses[x0] or crosses[y0];
        occ[x0] += occ[y0];
        alias[y0] = x0;
        coalesced += 1;
    };

    // Simplify.
    const removed = try gpa.alloc(bool, n);
    defer gpa.free(removed);
    const degree = try gpa.alloc(u32, n);
    defer gpa.free(degree);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(gpa);
    var left: usize = 0;
    for (0..n) |v| {
        removed[v] = iv[v].start == none or isFixed(f.fixed, v) or alias[v] != v;
        degree[v] = @intCast(adj[v].count());
        if (!removed[v]) left += 1;
    }
    while (left > 0) {
        var pick: ?usize = null;
        for (0..n) |v| if (!removed[v] and degree[v] < @popCount(allowed[v])) {
            pick = v;
            break;
        };
        if (pick == null) {
            // Optimistic: lowest weight = occ / (len + 1), compared by cross-multiplication.
            var best: ?usize = null;
            for (0..n) |v| if (!removed[v]) {
                if (best) |bv| {
                    const lv: u64 = iv[v].end - iv[v].start + 1;
                    const lb: u64 = iv[bv].end - iv[bv].start + 1;
                    if (@as(u64, occ[v]) * lb < @as(u64, occ[bv]) * lv) best = v;
                } else best = v;
            };
            pick = best;
        }
        const v = pick.?;
        removed[v] = true;
        left -= 1;
        try stack.append(gpa, @intCast(v));
        var it = adj[v].iterator(.{});
        while (it.next()) |w| degree[w] -|= 1;
    }

    // Select.
    var spilled: std.ArrayList(u32) = .empty;
    defer spilled.deinit(gpa);
    while (stack.pop()) |v| {
        var m = allowed[v];
        var it = adj[v].iterator(.{});
        while (it.next()) |w| if (locs[w] == .reg) {
            m &= ~(@as(RegMask, 1) << @intCast(locs[w].reg));
        };
        if (crosses[v] and m & cfg.callee_saved != 0) m &= cfg.callee_saved;
        if (m != 0) {
            locs[v] = .{ .reg = @intCast(@ctz(m)) };
        } else try spilled.append(gpa, v);
    }
    // Coalesced nodes take their representative's register (or are spilled with it).
    for (0..n) |v| if (alias[v] != v) {
        const rt = root(alias, @intCast(v));
        if (locs[rt] == .reg) locs[v] = locs[rt] else try spilled.append(gpa, @intCast(v));
    };
    // Stack slots: colour the spilled subgraph greedily (in vreg order).
    std.mem.sort(u32, spilled.items, {}, std.sort.asc(u32));
    var num_slots: u32 = 0;
    for (spilled.items) |v| {
        var slot: u32 = 0;
        search: while (true) : (slot += 1) {
            for (spilled.items) |w| {
                if (w == v or locs[w] != .stack or locs[w].stack != slot) continue;
                if (overlaps(iv[v], iv[w])) continue :search;
            }
            break;
        }
        locs[v] = .{ .stack = slot };
        num_slots = @max(num_slots, slot + 1);
    }
    const save = try saveSlots(gpa, f, cfg, iv, locs, calls.items, &num_slots);
    return .{ .locs = locs, .intervals = iv, .num_slots = num_slots, .spills = @intCast(spilled.items.len), .save_slot = save, .coalesced = coalesced, .entry_undef = undef };
}

pub fn allocateLinear(gpa: std.mem.Allocator, f: Func, cfg: Config) Error!Result {
    return allocateLinearInput(gpa, f, cfg);
}

fn allocateLinearInput(gpa: std.mem.Allocator, f: anytype, cfg: Config) Error!Result {
    var undef: []VReg = &.{};
    const iv = try buildIntervalsUndef(gpa, f, &undef);
    errdefer gpa.free(iv);
    errdefer if (undef.len != 0) gpa.free(undef);
    const locs = try gpa.alloc(Loc, f.num_vregs);
    errdefer gpa.free(locs);
    @memset(locs, .none);

    var calls = try callPositions(gpa, f);
    defer calls.deinit(gpa);

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
    var order = try std.ArrayList(Interval).initCapacity(gpa, f.num_vregs);
    defer order.deinit(gpa);
    for (iv, 0..) |x, v| {
        if (x.start == none) continue;
        if (f.fixed.len > v and f.fixed[v] != null) continue;
        order.appendAssumeCapacity(x);
    }
    std.mem.sort(Interval, order.items, {}, struct {
        fn lt(_: void, a: Interval, b: Interval) bool {
            return a.start < b.start or (a.start == b.start and a.vreg < b.vreg);
        }
    }.lt);

    // At most one active interval per physical register; no heap growth is needed.
    var active_buf: [@bitSizeOf(RegMask)]Interval = undefined;
    var active: std.ArrayList(Interval) = .initBuffer(&active_buf); // sorted by end
    var free_slots: std.ArrayList(SlotUse) = .empty;
    defer free_slots.deinit(gpa);
    var slot_in_use: std.ArrayList(SlotUse) = .empty;
    defer slot_in_use.deinit(gpa);
    var num_slots: u32 = 0;
    var spills: u32 = 0;
    var next_slot_end: u32 = none;

    for (order.items) |cur| {
        // Expire old intervals.
        var k: usize = 0;
        while (k < active.items.len) {
            if (active.items[k].end < cur.start) {
                _ = active.orderedRemove(k);
            } else k += 1;
        }
        if (next_slot_end < cur.start) {
            next_slot_end = none;
            k = 0;
            while (k < slot_in_use.items.len) {
                if (slot_in_use.items[k].end < cur.start) {
                    try free_slots.append(gpa, slot_in_use.swapRemove(k));
                } else {
                    next_slot_end = @min(next_slot_end, slot_in_use.items[k].end);
                    k += 1;
                }
            }
        }

        var mask = cfg.allocatable;
        const crosses = crossesCall(calls.items, cur);
        for (active.items) |a| if (locs[a.vreg] == .reg) {
            mask &= ~(@as(RegMask, 1) << @intCast(locs[a.vreg].reg));
        };
        for (fixed_iv.items, fixed_reg.items) |fi, r| if (overlaps(fi, cur)) {
            mask &= ~(@as(RegMask, 1) << @intCast(r));
        };

        if (crosses and (mask & cfg.callee_saved != 0 or !cfg.split_at_calls)) mask &= cfg.callee_saved;
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
            if (crosses and !cfg.split_at_calls and cfg.callee_saved & bit == 0) ok = false;
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
        next_slot_end = @min(next_slot_end, spilled.end);
        spills += 1;
    }
    // Split ranges: caller-saved registers holding call-crossing values get a save slot.
    const save = try gpa.alloc(?u32, f.num_vregs);
    @memset(save, null);
    for (iv, 0..) |x, v| {
        if (locs[v] != .reg or !crossesCall(calls.items, x)) continue;
        if (cfg.callee_saved & (@as(RegMask, 1) << @intCast(locs[v].reg)) != 0) continue;
        if (f.fixed.len > v and f.fixed[v] != null) continue;
        save[v] = num_slots;
        num_slots += 1;
    }
    return .{ .locs = locs, .intervals = iv, .num_slots = num_slots, .spills = spills, .save_slot = save, .entry_undef = undef };
}

fn root(alias: []u32, v: VReg) u32 {
    var x = v;
    while (alias[x] != x) x = alias[x];
    return x;
}

fn insertByEnd(gpa: std.mem.Allocator, list: *std.ArrayList(Interval), x: Interval) !void {
    var i: usize = 0;
    while (i < list.items.len and list.items[i].end <= x.end) i += 1;
    try list.insert(gpa, i, x);
}

/// Interference checker: overlapping intervals never share a register or a spill slot, and
/// intervals crossing a call only use callee-saved registers.
pub fn verify(f: Func, cfg: Config, r: Result) bool {
    return verifyInput(f, cfg, r);
}

pub fn allocateCompact(gpa: std.mem.Allocator, f: CompactFunc, cfg: Config) Error!Result {
    return allocateInput(gpa, f, cfg);
}

pub fn verifyCompact(f: CompactFunc, cfg: Config, r: Result) bool {
    return verifyInput(f, cfg, r);
}

fn verifyInput(f: anytype, cfg: Config, r: Result) bool {
    var pos: u32 = 0;
    for (f.blocks) |b| for (0..b.insts.len) |ii| {
        if (isCall(f, b, ii)) for (r.intervals) |x| {
            if (x.start == none) continue;
            if (x.start < 2 * pos and x.end > 2 * pos + 1) switch (r.locs[x.vreg]) {
                .reg => |reg| if (cfg.callee_saved & (@as(RegMask, 1) << @intCast(reg)) == 0) {
                    const is_fixed = f.fixed.len > x.vreg and f.fixed[x.vreg] != null;
                    if (!is_fixed and r.save_slot[x.vreg] == null) return false;
                },
                else => {},
            };
        };
        pos += 1;
    };
    for (r.segs, 0..) |sg, i| {
        if (r.locs[sg.vreg] != .stack or cfg.allocatable & (@as(RegMask, 1) << @intCast(sg.reg)) == 0) return false;
        const span: Interval = .{ .vreg = sg.vreg, .start = 2 * sg.first, .end = 2 * sg.last + 1 };
        for (r.intervals, 0..) |w, wi| if (w.start != none and overlaps(w, span) and r.locs[wi] == .reg and r.locs[wi].reg == sg.reg) return false;
        for (r.segs[i + 1 ..]) |o| if (o.reg == sg.reg and o.first <= sg.last and sg.first <= o.last) return false;
        var cp: u32 = 0;
        for (f.blocks) |b| for (0..b.insts.len) |ii| {
            if (isCall(f, b, ii) and (sg.first < cp or (sg.first == cp and sg.load)) and cp < sg.last and cfg.callee_saved & (@as(RegMask, 1) << @intCast(sg.reg)) == 0) return false;
            cp += 1;
        };
    }
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

test "regalloc: split at calls uses caller-saved registers with save slots" {
    // v0 defined, call, v0 used; only caller-saved registers r0, r1 exist.
    const blocks = [_]Block{.{ .insts = &.{
        .{ .defs = &.{0} },
        .{ .defs = &.{1}, .is_call = true },
        .{ .uses = &.{ 0, 1 } },
    } }};
    const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
    var plain = try allocate(t.allocator, f, .{ .allocatable = 0b11 });
    defer plain.deinit(t.allocator);
    try t.expect(plain.locs[0] == .stack); // no callee-saved: whole range spilled
    try t.expect(verify(f, .{ .allocatable = 0b11 }, plain));

    const cfg: Config = .{ .allocatable = 0b11, .split_at_calls = true };
    var r = try allocate(t.allocator, f, cfg);
    defer r.deinit(t.allocator);
    try t.expect(r.locs[0] == .reg);
    try t.expect(r.save_slot[0] != null);
    try t.expect(r.savedAcross(0, 1));
    try t.expect(!r.savedAcross(1, 1)); // the call's result is defined there
    try t.expect(verify(f, cfg, r));

    // A free callee-saved register is still preferred (no save needed).
    const cfg2: Config = .{ .allocatable = 0b111, .callee_saved = 0b100, .split_at_calls = true };
    var r2 = try allocate(t.allocator, f, cfg2);
    defer r2.deinit(t.allocator);
    try t.expectEqual(@as(u8, 2), r2.locs[0].reg);
    try t.expect(r2.save_slot[0] == null);
}

test "regalloc graph: same tests as linear scan" {
    // Loop, dying-source sharing, spill stress, calls, fixed registers.
    {
        const blocks = [_]Block{
            .{ .insts = &.{ .{ .defs = &.{0} }, .{ .defs = &.{1} } }, .succs = &.{1} },
            .{ .insts = &.{.{ .uses = &.{ 1, 0 }, .defs = &.{1} }}, .succs = &.{ 1, 2 } },
            .{ .insts = &.{.{ .uses = &.{1} }} },
        };
        const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
        const cfg: Config = .{ .allocatable = 0b11, .algo = .graph };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expectEqual(@as(u32, 0), r.spills);
    }
    {
        const blocks = [_]Block{.{ .insts = &.{ .{ .defs = &.{0} }, .{ .uses = &.{0}, .defs = &.{1} }, .{ .uses = &.{1} } } }};
        const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
        var r = try allocate(t.allocator, f, .{ .allocatable = 0b1, .algo = .graph });
        defer r.deinit(t.allocator);
        try t.expectEqual(@as(u32, 0), r.spills);
    }
    {
        const n = 50;
        var insts: [n + 1]Inst = undefined;
        var ids: [n]VReg = undefined;
        for (0..n) |i| {
            ids[i] = @intCast(i);
            insts[i] = .{ .defs = ids[i .. i + 1] };
        }
        insts[n] = .{ .uses = &ids };
        const blocks = [_]Block{.{ .insts = &insts }};
        const f: Func = .{ .blocks = &blocks, .num_vregs = n };
        const cfg: Config = .{ .allocatable = 0b111, .algo = .graph };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expectEqual(@as(u32, n - 3), r.spills);
        try t.expectEqual(@as(u32, n - 3), r.num_slots);
    }
    {
        const blocks = [_]Block{.{ .insts = &.{
            .{ .defs = &.{0} },
            .{ .defs = &.{1} },
            .{ .uses = &.{1}, .is_call = true },
            .{ .uses = &.{0} },
        } }};
        const f: Func = .{ .blocks = &blocks, .num_vregs = 2 };
        const cfg: Config = .{ .allocatable = 0b1111, .callee_saved = 0b1000, .algo = .graph };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expectEqual(Loc{ .reg = 3 }, r.locs[0]);
        const cfg_split: Config = .{ .allocatable = 0b11, .split_at_calls = true, .algo = .graph };
        var r2 = try allocate(t.allocator, f, cfg_split);
        defer r2.deinit(t.allocator);
        try t.expect(r2.locs[0] == .reg and r2.save_slot[0] != null);
        try t.expect(verify(f, cfg_split, r2));
    }
    {
        const blocks = [_]Block{.{ .insts = &.{ .{ .defs = &.{1} }, .{ .defs = &.{0} }, .{ .defs = &.{2} }, .{ .uses = &.{ 0, 1, 2 } } } }};
        const fixed = [_]?u8{ 1, null, null };
        const f: Func = .{ .blocks = &blocks, .num_vregs = 3, .fixed = &fixed };
        const cfg: Config = .{ .allocatable = 0b111, .algo = .graph };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expectEqual(Loc{ .reg = 1 }, r.locs[0]);
        try t.expect(r.locs[1].reg != 1 and r.locs[2].reg != 1);
        const bad = [_]?u8{ 1, 1, null };
        try t.expectError(error.FixedConflict, allocate(t.allocator, .{ .blocks = &blocks, .num_vregs = 3, .fixed = &bad }, cfg));
    }
}

test "regalloc graph: random programs verify; never more spills than registers allow" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var total_lin: u32 = 0;
    var total_graph: u32 = 0;
    for (0..300) |_| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const A = arena.allocator();
        const nv: u32 = 1 + rnd.uintLessThan(u32, 40);
        const nb: usize = 1 + rnd.uintLessThan(usize, 6);
        const blocks = try A.alloc(Block, nb);
        for (blocks) |*b| {
            const ni = 1 + rnd.uintLessThan(usize, 12);
            const insts = try A.alloc(Inst, ni);
            for (insts) |*in| {
                const u = try A.alloc(VReg, rnd.uintLessThan(usize, 3));
                for (u) |*x| x.* = rnd.uintLessThan(u32, nv);
                const d = try A.alloc(VReg, rnd.uintLessThan(usize, 2));
                for (d) |*x| x.* = rnd.uintLessThan(u32, nv);
                in.* = .{ .uses = u, .defs = d, .is_call = rnd.uintLessThan(u8, 8) == 0 };
            }
            const s = try A.alloc(u32, rnd.uintLessThan(usize, 3));
            for (s) |*x| x.* = rnd.uintLessThan(u32, @intCast(nb));
            b.* = .{ .insts = insts, .succs = s };
        }
        const f: Func = .{ .blocks = blocks, .num_vregs = nv };
        for ([_]bool{ false, true }) |split| {
            const cfg: Config = .{ .allocatable = 0b11111, .callee_saved = 0b11000, .split_at_calls = split, .algo = .graph };
            var r = try allocate(t.allocator, f, cfg);
            defer r.deinit(t.allocator);
            try t.expect(verify(f, cfg, r));
            var cl = cfg;
            cl.algo = .linear;
            var l = try allocate(t.allocator, f, cl);
            defer l.deinit(t.allocator);
            total_graph += r.spills;
            total_lin += l.spills;
        }
    }
    // Reported, not required to be lower (different heuristics).
    std.log.debug("spills: linear {d}, graph {d}", .{ total_lin, total_graph });
}

test "regalloc: splitting gives spilled values per-block register segments" {
    // v0..v3 all live across the block (4 values, 2 registers): two spill; each spilled value
    // is used twice in b1 while only one other value is live there => it gets a segment.
    const blocks = [_]Block{
        .{ .insts = &.{.{ .defs = &.{ 0, 1, 2, 3 } }}, .succs = &.{1} },
        .{ .insts = &.{ .{ .uses = &.{0} }, .{ .uses = &.{0} }, .{ .uses = &.{1} }, .{ .uses = &.{1} } }, .succs = &.{2} },
        .{ .insts = &.{ .{ .uses = &.{2} }, .{ .uses = &.{2} }, .{ .uses = &.{3} }, .{ .uses = &.{3} } } },
    };
    const f: Func = .{ .blocks = &blocks, .num_vregs = 4 };
    for ([_]Algo{ .linear, .graph }) |algo| {
        const cfg: Config = .{ .allocatable = 0b11, .algo = algo, .split = true };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expectEqual(@as(u32, 2), r.spills);
        try t.expect(r.segs.len >= 1);
        for (r.segs) |sg| {
            try t.expect(r.locs[sg.vreg] == .stack);
            try t.expect(sg.load); // first occurrence in the block is a use
            try t.expectEqual(@as(?u8, sg.reg), r.regAt(sg.vreg, sg.last));
        }
    }
}

test "regalloc graph: coalescing merges copy-related values" {
    // v0 = ..; v1 = copy v0; v2 = copy v1; use v2  => one register, two copies coalesced.
    const blocks = [_]Block{.{ .insts = &.{
        .{ .defs = &.{0} },
        .{ .uses = &.{0}, .defs = &.{1}, .is_copy = true },
        .{ .defs = &.{3} },
        .{ .uses = &.{1}, .defs = &.{2}, .is_copy = true },
        .{ .uses = &.{ 2, 3 } },
    } }};
    const f: Func = .{ .blocks = &blocks, .num_vregs = 4 };
    const cfg: Config = .{ .allocatable = 0b111, .algo = .graph };
    var r = try allocate(t.allocator, f, cfg);
    defer r.deinit(t.allocator);
    try t.expect(verify(f, cfg, r));
    try t.expectEqual(@as(u32, 2), r.coalesced);
    try t.expectEqual(r.locs[0], r.locs[1]);
    try t.expectEqual(r.locs[1], r.locs[2]);
    try t.expect(r.locs[3].reg != r.locs[2].reg);
    var r2 = try allocate(t.allocator, f, .{ .allocatable = 0b111, .algo = .graph, .coalesce = false });
    defer r2.deinit(t.allocator);
    try t.expectEqual(@as(u32, 0), r2.coalesced);
}

test "regalloc: random programs verify with splitting and coalescing" {
    var prng = std.Random.DefaultPrng.init(0x5917);
    const rnd = prng.random();
    for (0..300) |_| {
        const nv: u32 = 2 + rnd.uintLessThan(u32, 30);
        const nb: usize = 1 + rnd.uintLessThan(usize, 5);
        var blocks: [5]Block = undefined;
        var store: [5][20]Inst = undefined;
        var vstore: [5][20][2]VReg = undefined;
        var sstore: [5][2]u32 = undefined;
        for (0..nb) |bi| {
            const ni = 1 + rnd.uintLessThan(usize, 20);
            for (0..ni) |ii| {
                const a = rnd.uintLessThan(u32, nv);
                const d = rnd.uintLessThan(u32, nv);
                vstore[bi][ii] = .{ a, d };
                const kind = rnd.uintLessThan(u8, 4);
                store[bi][ii] = .{
                    .uses = if (kind == 0) &.{} else vstore[bi][ii][0..1],
                    .defs = if (kind == 3) &.{} else vstore[bi][ii][1..2],
                    .is_copy = kind == 1,
                    .is_call = kind == 2 and rnd.uintLessThan(u8, 3) == 0,
                };
            }
            sstore[bi] = .{ rnd.uintLessThan(u32, @intCast(nb)), rnd.uintLessThan(u32, @intCast(nb)) };
            blocks[bi] = .{ .insts = store[bi][0..ni], .succs = if (bi + 1 == nb) &.{} else sstore[bi][0..2] };
        }
        const f: Func = .{ .blocks = blocks[0..nb], .num_vregs = nv };
        for ([_]Algo{ .linear, .graph }) |algo| for ([_]bool{ false, true }) |sc| {
            const cfg: Config = .{ .allocatable = 0b1111, .callee_saved = 0b1100, .split_at_calls = sc, .algo = algo, .split = true };
            var r = try allocate(t.allocator, f, cfg);
            defer r.deinit(t.allocator);
            try t.expect(verify(f, cfg, r));
        };
    }
}

test "regalloc: a live-in value dying inside a block holds its register from the block start" {
    // Layout b0, b1, b2: v1 is defined in b2 (laid out after its use block b1) and dies in
    // b1, so it is not live-out of b1. v2/v3, defined at the top of b1, must not take its
    // register.
    const blocks = [_]Block{
        .{ .insts = &.{.{}}, .succs = &.{2} },
        .{ .insts = &.{ .{ .defs = &.{2} }, .{ .defs = &.{3} }, .{ .uses = &.{ 2, 3 }, .defs = &.{4} }, .{ .uses = &.{ 1, 4 }, .defs = &.{5} }, .{ .uses = &.{5} } } },
        .{ .insts = &.{ .{ .defs = &.{1} }, .{} }, .succs = &.{1} },
    };
    const f: Func = .{ .blocks = &blocks, .num_vregs = 6 };
    for ([_]Algo{ .linear, .graph }) |algo| {
        const cfg: Config = .{ .allocatable = 0b111, .algo = algo };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expect(r.intervals[1].start <= 2 * 1); // b1 starts at position 1
        for ([_]VReg{ 2, 3 }) |v| if (r.locs[v] == .reg and r.locs[1] == .reg) try t.expect(r.locs[v].reg != r.locs[1].reg);
    }
}

test "regalloc: a split segment loaded for a call argument and used after it is call-safe" {
    // b1: v2 = call(v0); use v0 v1 v2. With no callee-saved register, v0's segment starting
    // at the call must not sit in a (clobbered) caller-saved register.
    const blocks = [_]Block{
        .{ .insts = &.{ .{ .defs = &.{0} }, .{ .defs = &.{1} } }, .succs = &.{1} },
        .{ .insts = &.{ .{ .uses = &.{0}, .defs = &.{2}, .is_call = true }, .{ .uses = &.{ 0, 1, 2 } } } },
    };
    const f: Func = .{ .blocks = &blocks, .num_vregs = 3 };
    for ([_]Algo{ .linear, .graph }) |algo| {
        const cfg: Config = .{ .allocatable = 0b11, .callee_saved = 0, .split = true, .algo = algo };
        var r = try allocate(t.allocator, f, cfg);
        defer r.deinit(t.allocator);
        try t.expect(verify(f, cfg, r));
        try t.expect(r.regAt(0, 3) == null);
    }
    // verify rejects such a segment.
    var r = try allocate(t.allocator, f, .{ .allocatable = 0b11, .callee_saved = 0, .split = true });
    defer r.deinit(t.allocator);
    const bad = [_]Seg{.{ .vreg = 0, .first = 2, .last = 3, .reg = 1, .load = true }};
    const saved = r.segs;
    r.segs = &bad;
    defer r.segs = saved;
    if (r.locs[0] == .stack) try t.expect(!verify(f, .{ .allocatable = 0b11, .callee_saved = 0, .split = true }, r));
}
