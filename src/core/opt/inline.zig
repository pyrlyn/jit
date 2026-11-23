//! Runtime inlining (`Level.o2` with `opt.Options.resolver`): replaces `call` sites whose
//! target has IR (`Resolver.lookup`) by a copy of the callee's body, hottest sites first,
//! within the `opt.InlineOptions` budget. Runs before SSA construction, on mutable vregs;
//! block parameters may occur in the caller and in callees (they are cloned as they are).
//!
//! An inlined site becomes: typed zeros for the callee vregs that may be read before being
//! written (a real call starts with all registers 0; an inlined body in a loop would see the
//! previous iteration's values), `copy` of the arguments into the parameters, a branch into
//! the cloned blocks, and `ret v` -> `copy call.dst, v; br <tail>` where the tail block holds
//! the rest of the call's block.
const std = @import("std");
const ir = @import("../ir.zig");
const opt = @import("opt.zig");

/// A pending call site: `c.f.blocks[block].ops[index]` is a `call`.
const Site = struct {
    block: u32,
    index: u32,
    /// 1 for the caller's own calls, +1 per inlined body the call comes from.
    depth: u8,
    freq: ?u64,
};

/// Hotter sites first; then shallower, lower block, later in the block. All pending sites of
/// a block share its frequency and depth, so a block's sites are inlined back to front and
/// the positions of the remaining ones stay valid (the moved tail never holds pending sites).
fn hotterFirst(_: void, x: Site, y: Site) std.math.Order {
    const fx = x.freq orelse 0;
    const fy = y.freq orelse 0;
    if (fx != fy) return std.math.order(fy, fx);
    if (x.depth != y.depth) return std.math.order(x.depth, y.depth);
    if (x.block != y.block) return std.math.order(x.block, y.block);
    return std.math.order(y.index, x.index);
}

const Queue = std.PriorityQueue(Site, void, hotterFirst);

/// What inlining a callee needs (computed once per callee and run).
const Callee = struct {
    /// `g.typeCheck` (one entry per vreg).
    types: []ir.Type,
    /// Vregs that occur in `g` (results, operands, block parameters); only these get caller vregs.
    occurs: []bool,
    /// Vregs other than parameters that `g` may read before writing them on some path from its
    /// entry (a call reads them as 0), ascending.
    undef: []ir.VReg,
    /// Ops one inlining adds to the caller.
    cost: u32,

    fn deinit(info: *Callee, gpa: std.mem.Allocator) void {
        gpa.free(info.types);
        gpa.free(info.occurs);
        gpa.free(info.undef);
    }
};

/// Inlines call sites (see the module comment); true if any was inlined.
pub fn run(c: *opt.Ctx) opt.Error!bool {
    const r = c.options.resolver orelse return false;
    const lim = c.options.inlining;
    if (lim.max_depth == 0) return false;
    var queue: Queue = .empty;
    defer queue.deinit(c.gpa);
    for (c.f.blocks.items, 0..) |b, bi| for (b.ops.items, 0..) |op, i| if (op == .call)
        try queue.push(c.gpa, .{ .block = @intCast(bi), .index = @intCast(i), .depth = 1, .freq = c.freq.items[bi] });
    var callees: std.AutoHashMapUnmanaged(*const ir.Func, ?Callee) = .empty;
    defer {
        var it = callees.valueIterator();
        while (it.next()) |v| if (v.*) |*info| info.deinit(c.gpa);
        callees.deinit(c.gpa);
    }
    var map: std.ArrayList(ir.VReg) = .empty;
    defer map.deinit(c.gpa);
    var ops = opt.countOps(c.f);
    var changed = false;
    while (queue.pop()) |s| {
        if (ops >= lim.max_caller_ops) break;
        const call = c.f.blocks.items[s.block].ops.items[s.index].call;
        const g = r.lookup(r.ctx, call.addr) orelse continue;
        // Never the function being rewritten; never with the wrong arity (the native callee
        // would read undefined argument registers, the interpreter zeros).
        if (g == @as(*const ir.Func, c.f) or g.params != call.args.len) continue;
        const size = opt.countOps(g);
        const hot = if (s.freq) |q| q >= lim.hot_count else false;
        if (size > lim.max_callee_ops and !(hot and size <= lim.max_hot_callee_ops)) continue;
        const entry = try callees.getOrPut(c.gpa, g);
        if (!entry.found_existing) {
            entry.value_ptr.* = null;
            entry.value_ptr.* = try analyze(c.gpa, g);
        }
        const info = entry.value_ptr.* orelse continue;
        if (ops + info.cost > lim.max_caller_ops) continue;
        try map.resize(c.gpa, g.num_vregs);
        try inlineSite(c, s, g, &info, map.items, &queue);
        ops += info.cost;
        c.stats.inlined_calls += 1;
        changed = true;
    }
    return changed;
}

/// Callee facts for inlining, or null if `g` is not valid IR (such a callee is not inlined).
fn analyze(gpa: std.mem.Allocator, g: *const ir.Func) std.mem.Allocator.Error!?Callee {
    g.verify() catch return null;
    const types = g.typeCheck(gpa) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.TypeMismatch => return null,
    };
    errdefer gpa.free(types);
    const occurs = try gpa.alloc(bool, g.num_vregs);
    errdefer gpa.free(occurs);
    @memset(occurs, false);
    const undef = try maybeUndefined(gpa, g, occurs);
    var cost = opt.countOps(g) + @as(u32, @intCast(undef.len));
    for (g.blocks.items) |b| cost += @intFromBool(b.ops.getLast() == .ret); // ret -> copy + br
    for (occurs[0..g.params]) |o| cost += @intFromBool(o); // argument copies
    var lanes: [3]bool = @splat(false); // scalar zero for vector splats, per lane type
    for (undef) |v| if (types[v].isVector()) {
        lanes[laneIndex(types[v].lane())] = true;
    };
    for (lanes) |l| cost += @intFromBool(l);
    return .{ .types = types, .occurs = occurs, .undef = undef, .cost = cost };
}

fn laneIndex(lane: ir.Type) usize {
    return switch (lane) {
        .i64 => 0,
        .f32 => 1,
        else => 2,
    };
}

/// Vregs `>= g.params` that may be read before being written on some path from `g`'s entry
/// (backward liveness at the top of block 0; block parameters are assigned by incoming edges
/// but not at the function entry). Marks every vreg that occurs in `g` in `occurs`.
fn maybeUndefined(gpa: std.mem.Allocator, g: *const ir.Func, occurs: []bool) std.mem.Allocator.Error![]ir.VReg {
    const nb = g.blocks.items.len;
    const w = (g.num_vregs + 63) / 64;
    const sets = try gpa.alloc(u64, 3 * nb * w + 2 * w);
    defer gpa.free(sets);
    @memset(sets, 0);
    // Per block: upward-exposed reads, writes, live on entry with the block's parameters removed.
    const gen = sets[0 .. nb * w];
    const kill = sets[nb * w .. 2 * nb * w];
    const top = sets[2 * nb * w .. 3 * nb * w];
    const out = sets[3 * nb * w ..][0..w];
    const cur = sets[3 * nb * w + w ..][0..w];
    for (g.blocks.items, 0..) |b, bi| {
        const gb = gen[bi * w ..][0..w];
        const kb = kill[bi * w ..][0..w];
        for (b.params) |p| occurs[p] = true;
        for (b.ops.items) |op| {
            var buf: [ir.max_args]ir.VReg = undefined;
            const reads: [3][]const ir.VReg = switch (op) {
                .bra => |x| .{ x.args, &.{}, &.{} },
                .cbra => |x| blk: {
                    buf[0] = x.a;
                    buf[1] = x.b;
                    break :blk .{ buf[0..2], x.then_args, x.else_args };
                },
                else => .{ op.uses(&buf), &.{}, &.{} },
            };
            for (reads) |vs| for (vs) |v| {
                occurs[v] = true;
                if (!has(kb, v)) put(gb, v);
            };
            if (op.def()) |d| {
                occurs[d] = true;
                put(kb, d);
            }
        }
    }
    var changed = true;
    while (changed) {
        changed = false;
        var bi = nb;
        while (bi > 0) {
            bi -= 1;
            liveOut(g, top, w, bi, out);
            for (cur, gen[bi * w ..][0..w], kill[bi * w ..][0..w], out) |*x, gn, k, o| x.* = gn | (o & ~k);
            for (g.blocks.items[bi].params) |p| cur[p / 64] &= ~bit(p);
            const tb = top[bi * w ..][0..w];
            if (!std.mem.eql(u64, tb, cur)) {
                @memcpy(tb, cur);
                changed = true;
            }
        }
    }
    // Block 0 at the function entry: its parameters are not assigned.
    liveOut(g, top, w, 0, out);
    var undef: std.ArrayList(ir.VReg) = .empty;
    errdefer undef.deinit(gpa);
    for (0..g.num_vregs) |i| {
        const v: ir.VReg = @intCast(i);
        if (v >= g.params and (has(gen[0..w], v) or (has(out, v) and !has(kill[0..w], v)))) try undef.append(gpa, v);
    }
    return undef.toOwnedSlice(gpa);
}

/// `out` = union of `top` over the successors of block `bi`.
fn liveOut(g: *const ir.Func, top: []const u64, w: usize, bi: usize, out: []u64) void {
    @memset(out, 0);
    var buf: [2]u32 = undefined;
    const succs: []const u32 = switch (g.blocks.items[bi].ops.getLast()) {
        .br => |x| blk: {
            buf[0] = x;
            break :blk buf[0..1];
        },
        .bra => |x| blk: {
            buf[0] = x.target;
            break :blk buf[0..1];
        },
        inline .cbr, .cbra => |x| blk: {
            buf = .{ x.then, x.@"else" };
            break :blk buf[0..2];
        },
        else => buf[0..0],
    };
    for (succs) |s| for (out, top[s * w ..][0..w]) |*o, x| {
        o.* |= x;
    };
}

fn bit(v: ir.VReg) u64 {
    return @as(u64, 1) << @intCast(v % 64);
}
fn has(set: []const u64, v: ir.VReg) bool {
    return set[v / 64] & bit(v) != 0;
}
fn put(set: []u64, v: ir.VReg) void {
    set[v / 64] |= bit(v);
}

/// Inlines `g` at `s` (see the module comment); queues the calls of the cloned body.
fn inlineSite(c: *opt.Ctx, s: Site, g: *const ir.Func, info: *const Callee, map: []ir.VReg, queue: *Queue) opt.Error!void {
    const call = c.f.blocks.items[s.block].ops.items[s.index].call;
    // Blocks first (`addBlock` may move `c.f.blocks`): the tail, then one per callee block.
    const tail = try c.addBlock(c.f.blocks.items[s.block].name, s.freq);
    // The tail takes over the block's terminator, and with it the taken-edge count.
    c.taken.items[tail] = c.taken.items[s.block];
    c.taken.items[s.block] = null;
    const base: u32 = @intCast(c.f.blocks.items.len);
    for (g.blocks.items) |gb| _ = try c.addBlock(gb.name, s.freq);
    for (map, info.occurs, 0..) |*m, occ, v| m.* = if (occ) try c.newVReg(info.types[v]) else std.math.maxInt(ir.VReg);
    const a = c.irAlloc();
    const blocks = c.f.blocks.items;
    const ops = &blocks[s.block].ops;
    try blocks[tail].ops.appendSlice(a, ops.items[s.index + 1 ..]);
    ops.shrinkRetainingCapacity(s.index);
    var zero: [3]?ir.VReg = @splat(null);
    for (info.undef) |v| {
        const ty = info.types[v];
        if (!ty.isVector()) {
            try ops.append(a, zeroOp(ty, map[v]));
            continue;
        }
        const z = &zero[laneIndex(ty.lane())];
        if (z.* == null) {
            z.* = try c.newVReg(ty.lane());
            try ops.append(a, zeroOp(ty.lane(), z.*.?));
        }
        try ops.append(a, .{ .splat = .{ .ty = ty, .dst = map[v], .src = z.*.? } });
    }
    for (call.args, info.occurs[0..call.args.len], 0..) |arg, occ, p| if (occ)
        try ops.append(a, .{ .copy = .{ .dst = map[p], .src = arg } });
    const entry_params = g.blocks.items[0].params;
    try ops.append(a, if (entry_params.len == 0) .{ .br = base } else .{ .bra = .{ .target = base, .args = try mapped(a, entry_params, map) } });
    for (g.blocks.items, base..) |gb, nbi| {
        const nb = &blocks[nbi];
        nb.params = try mapped(a, gb.params, map);
        try nb.ops.ensureTotalCapacity(a, gb.ops.items.len + 1);
        for (gb.ops.items, 0..) |op, i| switch (op) {
            .ret => |v| nb.ops.appendSliceAssumeCapacity(&.{
                .{ .copy = .{ .dst = call.dst, .src = map[v] } },
                .{ .br = tail },
            }),
            else => {
                nb.ops.appendAssumeCapacity(try cloneOp(a, op, map, base));
                if (op == .call and s.depth < c.options.inlining.max_depth)
                    try queue.push(c.gpa, .{ .block = @intCast(nbi), .index = @intCast(i), .depth = s.depth + 1, .freq = s.freq });
            },
        };
    }
}

/// `dst = 0` of type `ty` (i64, f32 or f64; +0.0 for floats).
fn zeroOp(ty: ir.Type, dst: ir.VReg) ir.Op {
    return if (ty == .i64) .{ .@"const" = .{ .dst = dst, .imm = 0 } } else .{ .fconst = .{ .dst = dst, .ty = ty, .bits = 0 } };
}

/// `vs` renamed through `m`, in a new slice (callee memory is only borrowed).
fn mapped(a: std.mem.Allocator, vs: []const ir.VReg, m: []const ir.VReg) std.mem.Allocator.Error![]const ir.VReg {
    const out = try a.alloc(ir.VReg, vs.len);
    for (out, vs) |*o, v| o.* = m[v];
    return out;
}

/// Callee op `op` with every vreg (definition and uses) renamed through `m`, branch targets
/// offset by `base` and fresh argument slices. `ret` is handled by the caller.
fn cloneOp(a: std.mem.Allocator, op: ir.Op, m: []const ir.VReg, base: u32) std.mem.Allocator.Error!ir.Op {
    var o = op;
    switch (o) {
        .@"const", .fconst => {},
        inline .copy, .cvt, .splat, .extract => |*x| x.src = m[x.src],
        inline .arith, .typed, .fcmp => |*x| {
            x.a = m[x.a];
            x.b = m[x.b];
        },
        .load => |*x| x.base = m[x.base],
        .store => |*x| {
            x.base = m[x.base];
            x.src = m[x.src];
        },
        .call => |*x| x.args = try mapped(a, x.args, m),
        .br => |*x| x.* += base,
        .cbr => |*x| {
            x.a = m[x.a];
            x.b = m[x.b];
            x.then += base;
            x.@"else" += base;
        },
        .bra => |*x| {
            x.target += base;
            x.args = try mapped(a, x.args, m);
        },
        .cbra => |*x| {
            x.a = m[x.a];
            x.b = m[x.b];
            x.then += base;
            x.@"else" += base;
            x.then_args = try mapped(a, x.then_args, m);
            x.else_args = try mapped(a, x.else_args, m);
        },
        .ret => unreachable,
    }
    switch (o) {
        inline .@"const", .copy, .arith, .load, .call, .fconst, .typed, .fcmp, .cvt, .splat, .extract => |*x| x.dst = m[x.dst],
        else => {},
    }
    return o;
}

// ---- tests ------------------------------------------------------------------------------
// Callees are real `callconv(.c)` functions, so calls that stay calls still run in the
// interpreter: `Bound` functions interpret the IR the test resolver returns for them.
const t = std.testing;
const profile = @import("profile.zig");

/// Memory the test programs load from and store to.
var mem: [64]u64 = @splat(0);

const Bound = struct {
    const n = 4;
    var funcs: [n]?*const ir.Func = @splat(null);
    /// A native callee and its IR twin.
    var native: ?struct { addr: usize, f: *const ir.Func } = null;

    fn Tramp(comptime k: usize) type {
        return struct {
            fn call(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64, a11: i64, a12: i64, a13: i64, a14: i64, a15: i64) callconv(.c) i64 {
                const g = funcs[k].?;
                const all = [16]i64{ a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 };
                return ir.interpret(g, all[0..g.params], 1 << 22) catch @panic("bound callee failed");
            }
        };
    }
    fn addr(comptime k: usize) usize {
        return @intFromPtr(&Tramp(k).call);
    }
    fn addrs() [n]usize {
        var out: [n]usize = undefined;
        inline for (0..n) |k| out[k] = addr(k);
        return out;
    }
    fn lookup(_: ?*anyopaque, at: usize) ?*const ir.Func {
        inline for (0..n) |k| if (at == addr(k)) return funcs[k];
        if (native) |x| if (x.addr == at) return x.f;
        return null;
    }
    fn reset() void {
        funcs = @splat(null);
        native = null;
    }
};

/// Runs only the inliner on `f` (with structure and type checks), as `optimize` does.
fn inlineWith(f: *ir.Func, options: opt.Options) !opt.Stats {
    var o = options;
    o.resolver = .{ .lookup = Bound.lookup };
    o.verify_each = true;
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, f, &o, &stats);
    defer c.deinit();
    if (try run(&c)) try c.check("inline");
    return stats;
}

/// Interprets `want` and `got` on `args` from the same memory; results and memory must match.
fn expectSame(want: *const ir.Func, got: *const ir.Func, args: []const i64) !void {
    const init = mem;
    const r = try ir.interpret(want, args, 1 << 22);
    const after = mem;
    mem = init;
    try t.expectEqual(r, try ir.interpret(got, args, 1 << 22));
    try t.expectEqualSlices(u64, &after, &mem);
}

fn parseFmt(comptime src: []const u8, args: anytype) !ir.Func {
    const text = try std.fmt.allocPrint(t.allocator, src, args);
    defer t.allocator.free(text);
    return ir.parse(t.allocator, text);
}

fn countCalls(f: *const ir.Func) usize {
    var n: usize = 0;
    for (f.blocks.items) |b| for (b.ops.items) |op| {
        n += @intFromBool(op == .call);
    };
    return n;
}

/// Caller of `callee(i)` for `i < v0`, summing the results.
const loop_caller =
    \\func loop 1
    \\e:
    \\  v1 = const 0
    \\  v2 = const 0
    \\  v3 = const 1
    \\  br h
    \\h:
    \\  cbr lt v1 v0 body done
    \\body:
    \\  v4 = call {d} v1
    \\  v2 = add v2 v4
    \\  v1 = add v1 v3
    \\  br h
    \\done:
    \\  ret v2
;

test "inline: vregs read before written get typed zeros (i64, f64, vector) in a loop" {
    defer Bound.reset();
    // Reads before writes in a later block: v2 (written only on the other path) and v4.
    var late = try ir.parse(t.allocator,
        \\func late 1
        \\e:
        \\  v5 = const 3
        \\  v0 = sub v5 v0
        \\  v1 = const 0
        \\  cbr lt v0 v1 neg pos
        \\neg:
        \\  v3 = add v2 v0
        \\  ret v3
        \\pos:
        \\  v2 = add v0 v5
        \\  v3 = add v2 v4
        \\  v4 = copy v0
        \\  ret v3
    );
    defer late.deinit();
    Bound.funcs[1] = &late;
    var late_info = (try analyze(t.allocator, &late)).?;
    defer late_info.deinit(t.allocator);
    try t.expectEqualSlices(ir.VReg, &.{ 2, 4 }, late_info.undef);
    var want_late = try parseFmt(loop_caller, .{Bound.addr(1)});
    defer want_late.deinit();
    var got_late = try profile.clone(t.allocator, &want_late);
    defer got_late.deinit();
    try t.expectEqual(@as(u32, 1), (try inlineWith(&got_late, .{})).inlined_calls);
    for ([_]i64{ 0, 3, 4, 9 }) |n| try expectSame(&want_late, &got_late, &.{n});

    // v1, v5 and v8 are read before being written: 0 in every real call.
    var rbw = try ir.parse(t.allocator,
        \\func rbw 1
        \\e:
        \\  v2 = add v1 v0
        \\  v1 = mul v2 v0
        \\  v6 = fconst.f64 1.5
        \\  v5 = add.f64 v5 v6
        \\  v7 = cvt.i64.f64 v5
        \\  v9 = splat.v2i64 v0
        \\  v8 = add.v2i64 v8 v9
        \\  v10 = extract.v2i64 v8 1
        \\  v11 = add v1 v7
        \\  v11 = add v11 v10
        \\  ret v11
    );
    defer rbw.deinit();
    Bound.funcs[0] = &rbw;
    var info = (try analyze(t.allocator, &rbw)).?;
    defer info.deinit(t.allocator);
    try t.expectEqualSlices(ir.VReg, &.{ 1, 5, 8 }, info.undef);
    try t.expectEqual(opt.countOps(&rbw) + 3 + 1 + 1 + 1, info.cost); // zeros, lane zero, ret, v0
    var want = try parseFmt(loop_caller, .{Bound.addr(0)});
    defer want.deinit();
    var got = try profile.clone(t.allocator, &want);
    defer got.deinit();
    try t.expectEqual(@as(u32, 1), (try inlineWith(&got, .{})).inlined_calls);
    try t.expectEqual(@as(usize, 0), countCalls(&got));
    for ([_]i64{ 0, 1, 2, 5 }) |n| try expectSame(&want, &got, &.{n});
    try t.expectEqual(@as(i64, 30 + 5 + 10), try ir.interpret(&got, &.{5}, 1 << 20));
}

test "inline: callee with branches, a loop, memory and floats; two sites" {
    defer Bound.reset();
    var mix = try ir.parse(t.allocator,
        \\func mix 2
        \\e:
        \\  v2 = const 0
        \\  v3 = fconst.f64 0.0
        \\  v8 = const 8
        \\  v9 = const 1
        \\  br h
        \\h:
        \\  cbr lt v2 v1 body done
        \\body:
        \\  v4 = mul v2 v8
        \\  v5 = add v0 v4
        \\  v6 = load.f64 v5 0
        \\  v7 = fcmp.f64 lt v6 v3
        \\  cbr ne v7 v11 neg pos
        \\neg:
        \\  v6 = mul.f64 v6 v6
        \\  br pos
        \\pos:
        \\  v3 = add.f64 v3 v6
        \\  store.f64 v5 0 v3
        \\  v2 = add v2 v9
        \\  br h
        \\done:
        \\  v10 = cvt.i64.f64 v3
        \\  ret v10
    );
    defer mix.deinit();
    Bound.funcs[0] = &mix;
    var want = try parseFmt(
        \\func two 2
        \\e:
        \\  v2 = const {d}
        \\  v3 = call {d} v2 v0
        \\  v4 = call {d} v2 v1
        \\  v5 = mul v3 v4
        \\  v5 = add v5 v3
        \\  ret v5
    , .{ @intFromPtr(&mem), Bound.addr(0), Bound.addr(0) });
    defer want.deinit();
    var got = try profile.clone(t.allocator, &want);
    defer got.deinit();
    try t.expectEqual(@as(u32, 2), (try inlineWith(&got, .{})).inlined_calls);
    for (&mem, 0..) |*x, i| x.* = @bitCast(@as(f64, @floatFromInt(i)) * @as(f64, if (i % 3 == 0) -1.25 else 0.5));
    for ([_][2]i64{ .{ 0, 0 }, .{ 3, 5 }, .{ 9, 2 }, .{ 20, 30 } }) |a| try expectSame(&want, &got, &a);
}

test "inline: callees with block parameters, including on the entry block" {
    defer Bound.reset();
    // The entry block is a loop header with parameters (0 at the function entry).
    var cnt = try ir.parse(t.allocator,
        \\func cnt 1
        \\top(v1 v2):
        \\  v3 = const 1
        \\  v4 = add v2 v3
        \\  v5 = add v1 v0
        \\  cbr lt v4 v0 top:v5:v4 out:v5
        \\out(v6):
        \\  ret v6
    );
    defer cnt.deinit();
    var fib = try ir.parse(t.allocator, ir.fib_ssa_src);
    defer fib.deinit();
    Bound.funcs[0] = &cnt;
    Bound.funcs[1] = &fib;
    var want = try parseFmt(
        \\func both 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 0
        \\  v3 = const 1
        \\  br h v1
        \\h(v5):
        \\  cbr lt v5 v0 body:v5 done
        \\body(v6):
        \\  v4 = call {d} v6
        \\  v2 = add v2 v4
        \\  v4 = call {d} v6
        \\  v2 = xor v2 v4
        \\  v7 = add v6 v3
        \\  br h v7
        \\done:
        \\  ret v2
    , .{ Bound.addr(0), Bound.addr(1) });
    defer want.deinit();
    var got = try profile.clone(t.allocator, &want);
    defer got.deinit();
    try t.expectEqual(@as(u32, 2), (try inlineWith(&got, .{})).inlined_calls);
    for ([_]i64{ 0, 1, 4, 9 }) |n| try expectSame(&want, &got, &.{n});
}

test "inline: nested inlining stops at max_depth" {
    defer Bound.reset();
    var h = try ir.parse(t.allocator,
        \\func h 1
        \\e:
        \\  v1 = const 3
        \\  v2 = mul v0 v1
        \\  v2 = add v2 v3
        \\  v3 = const 1
        \\  ret v2
    );
    defer h.deinit();
    var g = try parseFmt(
        \\func g 1
        \\e:
        \\  v1 = call {d} v0
        \\  v2 = const 1
        \\  v3 = add v0 v2
        \\  v4 = call {d} v3
        \\  v5 = add v1 v4
        \\  ret v5
    , .{ Bound.addr(1), Bound.addr(1) });
    defer g.deinit();
    Bound.funcs[0] = &g;
    Bound.funcs[1] = &h;
    var want = try parseFmt(loop_caller, .{Bound.addr(0)});
    defer want.deinit();
    for ([_]u8{ 1, 2 }, [_]u32{ 1, 3 }, [_]usize{ 2, 0 }) |depth, inlined, calls_left| {
        var got = try profile.clone(t.allocator, &want);
        defer got.deinit();
        try t.expectEqual(inlined, (try inlineWith(&got, .{ .inlining = .{ .max_depth = depth } })).inlined_calls);
        try t.expectEqual(calls_left, countCalls(&got));
        for ([_]i64{ 0, 1, 6 }) |n| try expectSame(&want, &got, &.{n});
    }
}

test "inline: recursion unrolls at most max_depth levels" {
    defer Bound.reset();
    var rfib = try parseFmt(
        \\func rfib 1
        \\e:
        \\  v1 = const 2
        \\  cbr lt v0 v1 small big
        \\small:
        \\  ret v0
        \\big:
        \\  v2 = const 1
        \\  v3 = sub v0 v2
        \\  v4 = call {d} v3
        \\  v5 = sub v3 v2
        \\  v6 = call {d} v5
        \\  v7 = add v4 v6
        \\  ret v7
    , .{ Bound.addr(0), Bound.addr(0) });
    defer rfib.deinit();
    Bound.funcs[0] = &rfib;
    // Inlining into rfib itself: the resolver's IR is a different object than the caller.
    var got = try profile.clone(t.allocator, &rfib);
    defer got.deinit();
    try t.expectEqual(@as(u32, 2 + 4 + 8), (try inlineWith(&got, .{})).inlined_calls);
    try t.expectEqual(@as(usize, 16), countCalls(&got));
    for (0..16) |n| try expectSame(&rfib, &got, &.{@intCast(n)});
    try t.expectEqual(@as(i64, 610), try ir.interpret(&got, &.{15}, 1 << 20));
    // The function being optimized is never inlined into itself.
    Bound.funcs[0] = &got;
    const before = opt.countOps(&got);
    try t.expectEqual(@as(u32, 0), (try inlineWith(&got, .{})).inlined_calls);
    try t.expectEqual(before, opt.countOps(&got));
}

fn sub2(a: i64, b: i64) callconv(.c) i64 {
    return a -% b;
}

test "inline: arity mismatches and unknown targets stay calls" {
    defer Bound.reset();
    var sub_ir = try ir.parse(t.allocator,
        \\func sub2 2
        \\e:
        \\  v2 = sub v0 v1
        \\  ret v2
    );
    defer sub_ir.deinit();
    Bound.native = .{ .addr = @intFromPtr(&sub2), .f = &sub_ir };
    var want = try parseFmt(
        \\func m 2
        \\e:
        \\  v2 = call {d} v0
        \\  v3 = call {d} v0 v1 v1
        \\  v4 = call {d} v1 v0
        \\  v5 = add v2 v3
        \\  v5 = add v5 v4
        \\  ret v5
    , .{ @intFromPtr(&sub2), @intFromPtr(&sub2), Bound.addr(3) });
    defer want.deinit();
    Bound.funcs[3] = &sub_ir;
    var got = try profile.clone(t.allocator, &want);
    defer got.deinit();
    try t.expectEqual(@as(u32, 1), (try inlineWith(&got, .{})).inlined_calls);
    try t.expectEqual(@as(usize, 2), countCalls(&got));
    for ([_][2]i64{ .{ 7, 2 }, .{ -1, 1 << 62 } }) |a| try expectSame(&want, &got, &a);
    // No resolver: nothing to do.
    var stats: opt.Stats = .{};
    const o: opt.Options = .{ .verify_each = true };
    var c = try opt.Ctx.init(t.allocator, &got, &o, &stats);
    defer c.deinit();
    try t.expect(!try run(&c));
}

test "inline: callee size, caller budget and hottest-first order" {
    defer Bound.reset();
    var h = try ir.parse(t.allocator,
        \\func h 1
        \\e:
        \\  v1 = const 3
        \\  v2 = mul v0 v1
        \\  v3 = sub v2 v0
        \\  v4 = xor v3 v1
        \\  ret v4
    ); // 5 ops; inlining adds 7 (ret -> copy + br, argument copy)
    defer h.deinit();
    Bound.funcs[0] = &h;
    var want = try parseFmt(
        \\func two 1
        \\e:
        \\  cbr lt v0 v1 cold hot
        \\cold:
        \\  v2 = call {d} v0
        \\  ret v2
        \\hot:
        \\  v3 = call {d} v0
        \\  ret v3
    , .{ Bound.addr(0), Bound.addr(0) }); // 5 ops
    defer want.deinit();
    const freq = [_]u64{ 100, 1, 99 };
    const p: opt.Profile = .{ .blocks = &freq };
    const Case = struct { lim: opt.InlineOptions, prof: bool, inlined: u32, hot_left: bool, cold_left: bool };
    const cases = [_]Case{
        .{ .lim = .{ .max_callee_ops = 4 }, .prof = false, .inlined = 0, .hot_left = true, .cold_left = true },
        .{ .lim = .{ .max_callee_ops = 4, .max_hot_callee_ops = 5, .hot_count = 50 }, .prof = true, .inlined = 1, .hot_left = false, .cold_left = true },
        .{ .lim = .{ .max_callee_ops = 5, .max_caller_ops = 12 }, .prof = true, .inlined = 1, .hot_left = false, .cold_left = true },
        .{ .lim = .{ .max_callee_ops = 5, .max_caller_ops = 11 }, .prof = true, .inlined = 0, .hot_left = true, .cold_left = true },
        .{ .lim = .{ .max_callee_ops = 5, .max_caller_ops = 19 }, .prof = true, .inlined = 2, .hot_left = false, .cold_left = false },
        .{ .lim = .{ .max_depth = 0 }, .prof = false, .inlined = 0, .hot_left = true, .cold_left = true },
    };
    for (cases) |cs| {
        var got = try profile.clone(t.allocator, &want);
        defer got.deinit();
        const st = try inlineWith(&got, .{ .inlining = cs.lim, .profile = if (cs.prof) &p else null });
        try t.expectEqual(cs.inlined, st.inlined_calls);
        try t.expectEqual(cs.cold_left, got.blocks.items[1].ops.items[0] == .call);
        try t.expectEqual(cs.hot_left, got.blocks.items[2].ops.items[0] == .call);
        try t.expect(opt.countOps(&got) <= @max(opt.countOps(&want), cs.lim.max_caller_ops));
        for ([_]i64{ -5, 0, 5 }) |n| try expectSame(&want, &got, &.{n});
    }
}

test "inline: the taken count follows the terminator into the tail block" {
    defer Bound.reset();
    var h = try ir.parse(t.allocator,
        \\func h 1
        \\e:
        \\  v1 = add v0 v0
        \\  ret v1
    );
    defer h.deinit();
    Bound.funcs[0] = &h;
    var f = try parseFmt(
        \\func f 1
        \\e:
        \\  v1 = call {d} v0
        \\  cbr lt v1 v0 a b
        \\a:
        \\  ret v0
        \\b:
        \\  ret v1
    , .{Bound.addr(0)});
    defer f.deinit();
    const freq = [_]u64{ 10, 7, 3 };
    const taken = [_]u64{ 7, 0, 0 };
    const p: opt.Profile = .{ .blocks = &freq, .taken = &taken };
    var o: opt.Options = .{ .profile = &p, .resolver = .{ .lookup = Bound.lookup } };
    var stats: opt.Stats = .{};
    var c = try opt.Ctx.init(t.allocator, &f, &o, &stats);
    defer c.deinit();
    try t.expect(try run(&c));
    try t.expectEqual(@as(?u64, null), c.taken.items[0]);
    // Block 0 now jumps into the callee; the tail (first added block) ends in the cbr.
    const tail = 3;
    try t.expect(c.f.blocks.items[tail].ops.items[c.f.blocks.items[tail].ops.items.len - 1] == .cbr);
    try t.expectEqual(@as(?u64, 7), c.edgeFreq(tail, 0));
    try t.expectEqual(@as(?u64, 3), c.edgeFreq(tail, 1));
}

/// Random functions for differential tests: mutable vregs that are redefined and read before
/// being written, i64/f64 values, memory through `mem`, calls, block parameters, forward
/// branches and bounded loops (each back edge is taken at most `trips` times per call).
const RandGen = struct {
    rng: std.Random,
    f: *ir.Func,
    ints: std.ArrayList(ir.VReg) = .empty,
    floats: std.ArrayList(ir.VReg) = .empty,
    /// Written by an op that fixes its type (not only by copies or edges).
    defined: std.ArrayList(bool) = .empty,
    targets: []const usize,
    arity: []const u32,
    call_weight: u8,
    base: ir.VReg = 0,
    mask: ir.VReg = 0,
    one: ir.VReg = 0,

    const trips = 3;

    fn deinit(r: *RandGen) void {
        r.ints.deinit(t.allocator);
        r.floats.deinit(t.allocator);
        r.defined.deinit(t.allocator);
    }
    fn vreg(r: *RandGen) !ir.VReg {
        try r.defined.append(t.allocator, false);
        return r.f.newVReg();
    }
    fn fresh(r: *RandGen, ty: ir.Type) !ir.VReg {
        const v = try r.vreg();
        try (if (ty == .i64) &r.ints else &r.floats).append(t.allocator, v);
        return v;
    }
    fn pick(r: *RandGen, ty: ir.Type) ir.VReg {
        const pool = if (ty == .i64) r.ints.items else r.floats.items;
        return pool[r.rng.uintLessThan(usize, pool.len)];
    }
    fn dst(r: *RandGen, ty: ir.Type) !ir.VReg {
        return if (r.rng.boolean()) r.pick(ty) else r.fresh(ty);
    }
    fn emit(r: *RandGen, b: u32, o: ir.Op) !void {
        // A copy types its destination only if its source gets a type.
        if (o.def()) |d| if (o != .copy) {
            r.defined.items[d] = true;
        };
        try r.f.emit(b, o);
    }
    fn args(r: *RandGen, params: []const ir.VReg) ![]const ir.VReg {
        const out = try r.f.alloc().alloc(ir.VReg, params.len);
        for (out, params) |*o, p| o.* = r.pick(if (std.mem.indexOfScalar(ir.VReg, r.floats.items, p) != null) .f64 else .i64);
        return out;
    }
    /// Address of a random aligned `mem` slot in a new vreg.
    fn address(r: *RandGen, b: u32) !ir.VReg {
        const off = try r.vreg();
        const q = try r.vreg();
        try r.emit(b, .{ .arith = .{ .op = .@"and", .dst = off, .a = r.pick(.i64), .b = r.mask } });
        try r.emit(b, .{ .arith = .{ .op = .add, .dst = q, .a = r.base, .b = off } });
        return q;
    }
    fn op(r: *RandGen, b: u32) !void {
        const imms = [_]i64{ 0, 1, -1, 2, 7, 63, 64, -64, 1 << 40, std.math.maxInt(i64), std.math.minInt(i64) };
        const floats = [_]f64{ 0.0, -0.0, 1.5, -2.25, 1e300, std.math.nan(f64), std.math.inf(f64), 3.0 };
        const n = r.rng.uintLessThan(u8, 12 + r.call_weight);
        switch (if (n >= 12 and r.targets.len == 0) 2 else n) {
            0 => try r.emit(b, .{ .@"const" = .{ .dst = try r.dst(.i64), .imm = if (r.rng.boolean()) imms[r.rng.uintLessThan(usize, imms.len)] else r.rng.intRangeAtMost(i64, -9, 9) } }),
            1 => try r.emit(b, .{ .copy = .{ .dst = try r.dst(.i64), .src = r.pick(.i64) } }),
            2, 3 => try r.emit(b, .{ .arith = .{ .op = r.rng.enumValue(ir.Arith), .dst = try r.dst(.i64), .a = r.pick(.i64), .b = r.pick(.i64) } }),
            4 => {
                const ty: ir.Type = if (r.rng.boolean()) .i64 else .f64;
                const q = try r.address(b);
                try r.emit(b, .{ .load = .{ .dst = try r.dst(ty), .base = q, .disp = 0, .ty = ty } });
            },
            5 => {
                const ty: ir.Type = if (r.rng.boolean()) .i64 else .f64;
                const q = try r.address(b);
                try r.emit(b, .{ .store = .{ .base = q, .disp = 0, .src = r.pick(ty), .ty = ty } });
            },
            6 => try r.emit(b, .{ .fconst = .{ .dst = try r.dst(.f64), .ty = .f64, .bits = @bitCast(floats[r.rng.uintLessThan(usize, floats.len)]) } }),
            7 => {
                const ops = [_]ir.TArith{ .add, .sub, .mul, .div, .min, .max };
                try r.emit(b, .{ .typed = .{ .op = ops[r.rng.uintLessThan(usize, ops.len)], .ty = .f64, .dst = try r.dst(.f64), .a = r.pick(.f64), .b = r.pick(.f64) } });
            },
            8 => try r.emit(b, .{ .fcmp = .{ .cond = r.rng.enumValue(ir.FCond), .ty = .f64, .dst = try r.dst(.i64), .a = r.pick(.f64), .b = r.pick(.f64) } }),
            9 => try r.emit(b, .{ .cvt = .{ .to = .f64, .from = .i64, .dst = try r.dst(.f64), .src = r.pick(.i64) } }),
            10 => try r.emit(b, .{ .cvt = .{ .to = .i64, .from = .f64, .dst = try r.dst(.i64), .src = r.pick(.f64) } }),
            11 => try r.emit(b, .{ .copy = .{ .dst = try r.dst(.f64), .src = r.pick(.f64) } }),
            else => {
                const k = r.rng.uintLessThan(usize, r.targets.len);
                // Mostly the right arity; sometimes not (must stay a call).
                const na = if (r.rng.uintLessThan(u8, 8) == 0) r.rng.uintLessThan(u32, 4) else r.arity[k];
                const a = try r.f.alloc().alloc(ir.VReg, na);
                for (a) |*x| x.* = r.pick(.i64);
                try r.emit(b, .{ .call = .{ .dst = try r.dst(.i64), .addr = r.targets[k], .args = a } });
            },
        }
    }

    /// Edge from `b` to `target` (with arguments when it has parameters).
    fn edge(r: *RandGen, target: u32) !struct { u32, []const ir.VReg } {
        return .{ target, try r.args(r.f.blocks.items[target].params) };
    }
    fn cond(r: *RandGen, b: u32, cc: ir.CondCode, x: ir.VReg, y: ir.VReg, th: u32, el: u32) !void {
        const tt, const ta = try r.edge(th);
        const et, const ea = try r.edge(el);
        try r.emit(b, if (ta.len == 0 and ea.len == 0)
            .{ .cbr = .{ .cond = cc, .a = x, .b = y, .then = tt, .@"else" = et } }
        else
            .{ .cbra = .{ .cond = cc, .a = x, .b = y, .then = tt, .@"else" = et, .then_args = ta, .else_args = ea } });
    }

    fn gen(rng: std.Random, params: u32, targets: []const usize, arity: []const u32, call_weight: u8) !ir.Func {
        var f = ir.Func.init(t.allocator);
        errdefer f.deinit();
        var r: RandGen = .{ .rng = rng, .f = &f, .targets = targets, .arity = arity, .call_weight = call_weight };
        defer r.deinit();
        for (0..params) |_| _ = try r.fresh(.i64);
        f.params = params;
        _ = try r.fresh(.i64); // never written: reads 0
        _ = try r.fresh(.f64); // written only at the end: read before written
        const nb = 1 + rng.uintLessThan(u32, 6);
        var name: [8]u8 = undefined;
        for (0..nb) |i| _ = try f.addBlock(std.fmt.bufPrint(&name, "b{d}", .{i}) catch unreachable);
        for (1..nb) |i| if (rng.uintLessThan(u8, 3) == 0) {
            const ps = try f.alloc().alloc(ir.VReg, 1 + rng.uintLessThan(usize, 2));
            for (ps) |*p| p.* = try r.fresh(if (rng.boolean()) .i64 else .f64);
            f.blocks.items[i].params = ps;
        };
        // Terminator kinds: 0 jump, 1 conditional, 2 bounded back edge, 3 early return.
        var kinds: [6]u8 = undefined;
        var others: [6]u32 = undefined; // second target of a conditional
        var reached: [6]bool = @splat(false);
        var ctrs: [6]ir.VReg = undefined;
        r.base = try r.vreg();
        r.mask = try r.vreg();
        r.one = try r.vreg();
        const limit = try r.vreg();
        try r.emit(0, .{ .@"const" = .{ .dst = r.base, .imm = @bitCast(@as(u64, @intFromPtr(&mem))) } });
        try r.emit(0, .{ .@"const" = .{ .dst = r.mask, .imm = (mem.len - 1) * 8 } });
        try r.emit(0, .{ .@"const" = .{ .dst = r.one, .imm = 1 } });
        try r.emit(0, .{ .@"const" = .{ .dst = limit, .imm = trips } });
        for (0..nb - 1) |i| {
            kinds[i] = rng.uintLessThan(u8, 4);
            if (kinds[i] == 2 and i == 0) kinds[i] = 1;
            if (kinds[i] == 3 and !reached[i + 1]) kinds[i] = 0;
            if (kinds[i] != 3) reached[i + 1] = true;
            others[i] = @intCast(i + 1 + rng.uintLessThan(usize, nb - 1 - i));
            if (kinds[i] == 1) reached[others[i]] = true;
            if (kinds[i] == 2) {
                ctrs[i] = try r.vreg();
                try r.emit(0, .{ .@"const" = .{ .dst = ctrs[i], .imm = 0 } });
            }
        }
        for (0..nb) |bi| {
            const b: u32 = @intCast(bi);
            for (0..rng.uintLessThan(u8, 6)) |_| try r.op(b);
            if (b == nb - 1 or kinds[b] == 3) {
                if (b == nb - 1) for (r.floats.items) |v| if (!r.defined.items[v])
                    try r.emit(b, .{ .fconst = .{ .dst = v, .ty = .f64, .bits = @bitCast(@as(f64, 0.75)) } });
                // The result depends on several values (stale values would show).
                const x = try r.vreg();
                const y = try r.vreg();
                try r.emit(b, .{ .cvt = .{ .to = .i64, .from = .f64, .dst = y, .src = r.pick(.f64) } });
                try r.emit(b, .{ .arith = .{ .op = .xor, .dst = x, .a = r.pick(.i64), .b = y } });
                for (0..3) |_| try r.emit(b, .{ .arith = .{ .op = .add, .dst = x, .a = x, .b = r.pick(.i64) } });
                try r.emit(b, .{ .ret = x });
                continue;
            }
            const next = b + 1;
            switch (kinds[b]) {
                0 => {
                    const tg, const a = try r.edge(next);
                    try r.emit(b, if (a.len == 0) .{ .br = tg } else .{ .bra = .{ .target = tg, .args = a } });
                },
                1 => {
                    const th, const el = if (rng.boolean()) .{ next, others[b] } else .{ others[b], next };
                    try r.cond(b, rng.enumValue(ir.CondCode), r.pick(.i64), r.pick(.i64), th, el);
                },
                else => {
                    try r.emit(b, .{ .arith = .{ .op = .add, .dst = ctrs[b], .a = ctrs[b], .b = r.one } });
                    try r.cond(b, .le, ctrs[b], limit, 1 + rng.uintLessThan(u32, b), next);
                },
            }
        }
        return f;
    }
};

test "inline: random callees and callers agree with the interpreter" {
    defer Bound.reset();
    const targets = Bound.addrs();
    var inlined: u32 = 0;
    for (0..300) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var callees: [Bound.n]ir.Func = undefined;
        var arity: [Bound.n]u32 = undefined;
        var made: usize = 0;
        defer for (callees[0..made]) |*g| g.deinit();
        // Callee k calls callees below k only (no recursion).
        for (0..Bound.n) |k| {
            callees[k] = try RandGen.gen(rng, rng.uintLessThan(u32, 4), targets[0..k], arity[0..k], 2);
            made += 1;
            arity[k] = callees[k].params;
            Bound.funcs[k] = &callees[k];
        }
        var want = try RandGen.gen(rng, 2, &targets, &arity, 8);
        defer want.deinit();
        const freq = try t.allocator.alloc(u64, want.blocks.items.len);
        defer t.allocator.free(freq);
        for (freq) |*q| q.* = rng.uintLessThan(u64, 100);
        const p: opt.Profile = .{ .blocks = freq };
        const o: opt.Options = .{
            .inlining = .{
                .max_callee_ops = rng.uintLessThan(u32, 80),
                .max_hot_callee_ops = rng.uintLessThan(u32, 160),
                .hot_count = 1 + rng.uintLessThan(u64, 99),
                .max_caller_ops = if (rng.boolean()) 4000 else rng.uintLessThan(u32, 300),
                .max_depth = rng.uintLessThan(u8, 4),
            },
            .profile = if (rng.boolean()) &p else null,
        };
        var got = try profile.clone(t.allocator, &want);
        defer got.deinit();
        inlined += (try inlineWith(&got, o)).inlined_calls;
        // The whole pipeline too (verify_each), on another copy.
        var full = try profile.clone(t.allocator, &want);
        defer full.deinit();
        var fo = o;
        fo.resolver = .{ .lookup = Bound.lookup };
        fo.verify_each = true;
        _ = try opt.optimize(t.allocator, &full, fo);
        for (0..3) |_| {
            for (&mem) |*x| x.* = if (rng.boolean()) rng.int(u64) else rng.uintLessThan(u64, 16);
            const a = [2]i64{ rng.intRangeAtMost(i64, -3, 12), rng.int(i64) };
            const init = mem;
            try expectSame(&want, &got, &a);
            mem = init;
            try expectSame(&want, &full, &a);
        }
    }
    try t.expect(inlined > 300);
}
