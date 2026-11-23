const std = @import("std");
const h = @import("vec_native_tests.zig");
const t = std.testing;
const gpa = std.heap.page_allocator;
const minI = std.math.minInt(i64);
const maxI = std.math.maxInt(i64);

const rels = [_][]const u8{ "lt", "ult", "le", "ule", "ne" };
fn negate(r: []const u8) []const u8 {
    const m = .{ .{ "lt", "ge" }, .{ "ult", "uge" }, .{ "le", "gt" }, .{ "ule", "ugt" }, .{ "ne", "eq" } };
    inline for (m) |p| if (std.mem.eql(u8, r, p[0])) return p[1];
    unreachable;
}
fn mirror(r: []const u8) []const u8 {
    const m = .{ .{ "lt", "gt" }, .{ "ult", "ugt" }, .{ "le", "ge" }, .{ "ule", "uge" }, .{ "ne", "ne" } };
    inline for (m) |p| if (std.mem.eql(u8, r, p[0])) return p[1];
    unreachable;
}

/// Exit test text: continue iff x REL n, in 3 spellings.
fn testText(buf: []u8, form: u8, rel: []const u8, x: []const u8, n: []const u8, cont: []const u8, exit: []const u8) ![]u8 {
    return switch (form) {
        0 => std.fmt.bufPrint(buf, "cbr {s} {s} {s} {s} {s}", .{ rel, x, n, cont, exit }),
        1 => std.fmt.bufPrint(buf, "cbr {s} {s} {s} {s} {s}", .{ negate(rel), x, n, exit, cont }),
        else => std.fmt.bufPrint(buf, "cbr {s} {s} {s} {s} {s}", .{ mirror(rel), n, x, cont, exit }),
    };
}

/// func f 3: v0 = array, v1 = start, v2 = n. a[(i - s) * 8/S] += 8 (also summed), i += S.
fn tripSrc(top: bool, use_next: bool, form: u8, rel: []const u8, step: i64, off: i64) ![]u8 {
    var tb: [128]u8 = undefined;
    const scale = @divExact(8, step);
    if (top) {
        const tt = try testText(&tb, form, rel, "v10", "v2", "body", "done");
        return std.fmt.allocPrint(gpa,
            \\func trip 3
            \\e:
            \\  v3 = copy v1
            \\  v4 = const 8
            \\  v5 = const {d}
            \\  v11 = const {d}
            \\  v12 = const {d}
            \\  v13 = const 0
            \\  br head
            \\head:
            \\  v10 = add v3 v11
            \\  {s}
            \\body:
            \\  v6 = sub v3 v1
            \\  v6 = mul v6 v12
            \\  v7 = add v0 v6
            \\  v8 = load v7 0
            \\  v13 = add v13 v8
            \\  v9 = add v8 v4
            \\  store v7 0 v9
            \\  v3 = add v3 v5
            \\  br head
            \\done:
            \\  v14 = mul v13 v4
            \\  v14 = xor v14 v3
            \\  ret v14
        , .{ step, off, scale, tt });
    }
    const tt = try testText(&tb, form, rel, "v10", "v2", "loop", "done");
    return std.fmt.allocPrint(gpa,
        \\func trip 3
        \\e:
        \\  v3 = copy v1
        \\  v4 = const 8
        \\  v5 = const {d}
        \\  v11 = const {d}
        \\  v12 = const {d}
        \\  v13 = const 0
        \\  br loop
        \\loop:
        \\  v6 = sub v3 v1
        \\  v6 = mul v6 v12
        \\  v7 = add v0 v6
        \\  v8 = load v7 0
        \\  v13 = add v13 v8
        \\  v9 = add v8 v4
        \\  store v7 0 v9
        \\  v15 = copy v3
        \\  v3 = add v3 v5
        \\  v10 = add {s} v11
        \\  {s}
        \\done:
        \\  v14 = mul v13 v4
        \\  v14 = xor v14 v3
        \\  v14 = add v14 v10
        \\  v14 = add v14 v6
        \\  ret v14
    , .{ step, off, scale, if (use_next) "v3" else "v15", tt });
}

test "vecnative: trip-count guards" {
    const starts = [_]i64{ 0, -3, minI, maxI - 3, maxI, -1 };
    const deltas = [_]i64{ -2, -1, 0, 1, 2, 3, 4, 5, 8, 9, 16, 17 };
    const extra = [_]i64{ minI, maxI, 0, -1, 1, minI + 1, maxI - 1 };
    const offs = [_]i64{ 0, 3, maxI, minI };
    var fails: usize = 0;
    var programs: usize = 0;
    for ([_]bool{ false, true }) |top| for ([_]bool{ false, true }) |use_next| for (0..3) |form| for (rels) |rel| for ([_]i64{ 1, 2, 8 }) |step| for (offs) |off| {
        if (top and use_next) continue;
        const src = try tripSrc(top, use_next, @intCast(form), rel, step, off);
        defer gpa.free(src);
        var sets: std.ArrayList([3]i64) = .empty;
        defer sets.deinit(gpa);
        for (starts) |s| {
            for (deltas) |d| for ([_]i64{ 0, 1 }) |mis| {
                // x at iteration j is s + off + j*S (top: current; bottom: depends)
                try sets.append(gpa, .{ 1024, s, s +% off +% d *% step +% mis });
            };
            for (extra) |n| try sets.append(gpa, .{ 1024, s, n });
        }
        var slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        programs += 1;
        h.check(src, 1, slices, .i64, .{ .min_vec = 0 }) catch |e| {
            fails += 1;
            std.debug.print("FAIL ({s}) top={} next={} form={d} rel={s} step={d} off={d}\n", .{ @errorName(e), top, use_next, form, rel, step, off });
            if (fails > 6) return e;
        };
    };
    std.debug.print("trip programs {d}, fails {d}, vectorized {d}, runs {d}\n", .{ programs, fails, h.vec_count, h.run_count });
    try t.expectEqual(@as(usize, 0), fails);
}

fn runSets(src: []const u8, ptrs: u8, sets: []const []const i64, fill: h.Fill, o: h.Opt) !void {
    h.check(src, ptrs, sets, fill, o) catch |e| {
        std.debug.print("FAILED SRC:\n{s}\n", .{src});
        return e;
    };
}

test "vecnative: alias checks" {
    var fails: usize = 0;
    // ty, elem size
    const tys = [_]struct { []const u8, i64, h.Fill }{ .{ "", 8, .i64 }, .{ ".f64", 8, .f64 }, .{ ".f32", 4, .f32 } };
    for (tys) |ty| for ([_]i64{ 0, 16, -16, 32 }) |d1| for ([_]i64{ 0, 8, -8, 4, -4 }) |d2| {
        const src = try std.fmt.allocPrint(gpa,
            \\func alias 3
            \\e:
            \\  v4 = const 0
            \\  v5 = const {d}
            \\  v6 = const 1
            \\  cbr ge v4 v2 done loop
            \\loop:
            \\  v7 = mul v4 v5
            \\  v8 = add v0 v7
            \\  v9 = add v1 v7
            \\  v10 = load{s} v9 {d}
            \\  v11 = load{s} v8 {d}
            \\  v12 = add{s} v10 v11
            \\  store{s} v8 0 v12
            \\  v4 = add v4 v6
            \\  cbr lt v4 v2 loop done
            \\done:
            \\  ret v4
        , .{ ty[1], ty[0], d2, ty[0], d1, ty[0], ty[0] });
        defer gpa.free(src);
        var sets: std.ArrayList([3]i64) = .empty;
        defer sets.deinit(gpa);
        var off: i64 = -72;
        while (off <= 72) : (off += 1) for ([_]i64{ 0, 1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 31 }) |n| {
            try sets.append(gpa, .{ 2048, 2048 + off, n });
        };
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src, 3, slices, ty[2], .{}) catch |e| {
            fails += 1;
            std.debug.print("alias FAIL {s} ty={s} d1={d} d2={d}\n", .{ @errorName(e), ty[0], d1, d2 });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: reductions" {
    var fails: usize = 0;
    var mul_t = @import("core/opt/opt.zig").VectorTarget.host;
    mul_t.mul_v2i64 = true;
    for ([_][]const u8{ "add", "sub", "and", "or", "xor", "mul" }) |op| for ([_]bool{ false, true }) |uni| for ([_]bool{ false, true }) |swap| {
        if (swap and std.mem.eql(u8, op, "sub")) continue;
        const x = if (uni) "v2" else "v9";
        const src = try std.fmt.allocPrint(gpa,
            \\func red 3
            \\e:
            \\  v3 = const 0
            \\  v4 = const 8
            \\  v5 = const 1
            \\  v6 = copy v2
            \\  v11 = const 7
            \\  br loop
            \\loop:
            \\  v7 = mul v3 v4
            \\  v8 = add v0 v7
            \\  v9 = load v8 0
            \\  v9 = or v9 v5
            \\  v6 = {s} {s} {s}
            \\  v11 = xor v11 v9
            \\  v3 = add v3 v5
            \\  cbr lt v3 v1 loop done
            \\done:
            \\  v12 = mul v11 v4
            \\  v12 = add v12 v6
            \\  ret v12
        , .{ op, if (swap) x else "v6", if (swap) "v6" else x });
        defer gpa.free(src);
        var sets: std.ArrayList([3]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 33, -5, minI }) |n| for ([_]i64{ 0, -1, 3, minI, 0x123456789 }) |init| {
            try sets.append(gpa, .{ 1024, n, init });
        };
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        const is_mul = std.mem.eql(u8, op, "mul");
        runSets(src, 1, slices, .i64, .{ .target = if (is_mul) mul_t else @import("core/opt/opt.zig").VectorTarget.host }) catch |e| {
            fails += 1;
            std.debug.print("red FAIL {s} op={s} uni={} swap={}\n", .{ @errorName(e), op, uni, swap });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: pointer IVs" {
    var fails: usize = 0;
    for ([_][]const u8{ "ult", "ne", "ule" }) |rel| for ([_]i64{ 0, 16, -16, 32 }) |d| {
        const src = try std.fmt.allocPrint(gpa,
            \\func pivs 3
            \\e:
            \\  v3 = copy v0
            \\  v4 = copy v2
            \\  v5 = const 8
            \\  br loop
            \\loop:
            \\  v6 = load v3 0
            \\  v7 = load v4 {d}
            \\  v8 = sub v6 v7
            \\  store v4 0 v8
            \\  v3 = add v3 v5
            \\  v4 = add v4 v5
            \\  cbr {s} v3 v1 loop done
            \\done:
            \\  v9 = sub v3 v0
            \\  v10 = sub v4 v2
            \\  v9 = mul v9 v5
            \\  v9 = xor v9 v10
            \\  ret v9
        , .{ d, rel });
        defer gpa.free(src);
        var sets: std.ArrayList([3]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 8, 16, 24, 32, 40, 64, 72, 136, 3, 5, 12, 17, -8, -16 }) |len| for ([_]i64{ 0, 8, -8, 16, -16, 4096, 24, -24, 3 }) |bo| {
            try sets.append(gpa, .{ 8192, 8192 + len, 8192 + bo + d });
        };
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src, 7, slices, .i64, .{}) catch |e| {
            fails += 1;
            std.debug.print("piv FAIL {s} rel={s} d={d}\n", .{ @errorName(e), rel, d });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: float lane ops (min/max/div, splats) vs interpreter" {
    var fails: usize = 0;
    for ([_]struct { []const u8, i64, h.Fill }{ .{ "f32", 4, .f32s }, .{ "f64", 8, .f64s }, .{ "f32", 4, .f32 }, .{ "f64", 8, .f64 } }) |ty| for ([_][]const u8{ "add", "sub", "mul", "div", "min", "max" }) |op| for (0..4) |form| {
        // form 0: op(x, y); 1: op(y, x); 2: op(x, k); 3: op(k, x)
        const a, const b = switch (form) {
            0 => .{ "v10", "v11" },
            1 => .{ "v11", "v10" },
            2 => .{ "v10", "v20" },
            else => .{ "v20", "v10" },
        };
        const src = try std.fmt.allocPrint(gpa,
            \\func fl 5
            \\e:
            \\  v21 = const 0
            \\  v5 = const {d}
            \\  v6 = const 1
            \\  v20 = load.{s} v4 0
            \\  cbr ge v21 v3 done loop
            \\loop:
            \\  v7 = mul v21 v5
            \\  v8 = add v0 v7
            \\  v9 = add v1 v7
            \\  v15 = add v2 v7
            \\  v10 = load.{s} v8 0
            \\  v11 = load.{s} v9 0
            \\  v12 = {s}.{s} {s} {s}
            \\  store.{s} v15 0 v12
            \\  v21 = add v21 v6
            \\  cbr lt v21 v3 loop done
            \\done:
            \\  ret v21
        , .{ ty[1], ty[0], ty[0], ty[0], op, ty[0], a, b, ty[0] });
        defer gpa.free(src);
        // v20 is loaded from address 0 -> replace: use v4? fixed below
        const src2 = try std.mem.replaceOwned(u8, gpa, src, "v20 = load.", "v20 = load.");
        defer gpa.free(src2);
        var sets: std.ArrayList([5]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 64 }) |n| for ([_]i64{ 0, 8, 16, 40, 4, 12 }) |bo| {
            try sets.append(gpa, .{ 1024, 2048 + bo, 8192, n, 12000 + 4 * bo });
        };
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src2, 7 | 16, slices, ty[2], .{}) catch |e| {
            fails += 1;
            std.debug.print("float FAIL {s} ty={s} op={s} form={d}\n", .{ @errorName(e), ty[0], op, form });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

/// N loads of one array at disps 16k, all live at once, combined and stored; R xor/add reductions.
fn pressureSrc(ty: []const u8, esz: i64, nl: usize, nr: usize, ops: []const []const u8) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    try b.print(gpa, "func press 2\ne:\n  v2 = const 0\n  v3 = const {d}\n  v4 = const 1\n", .{esz});
    for (0..nr) |r| try b.print(gpa, "  v{d} = const {d}\n", .{ 300 + r, r * 7 + 1 });
    try b.print(gpa, "  cbr ge v2 v1 done loop\nloop:\n  v5 = mul v2 v3\n  v6 = add v0 v5\n", .{});
    for (0..nl) |k| try b.print(gpa, "  v{d} = load{s} v6 {d}\n", .{ 100 + k, ty, 16 * k });
    // reductions over i64 loads (only when ty is i64)
    for (0..nr) |r| try b.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ 300 + r, if (r % 2 == 0) "xor" else "add", 300 + r, 100 + (r % nl) });
    // combine in reverse so everything stays live
    var acc: usize = 100 + nl - 1;
    var k: usize = nl - 1;
    var tmp: usize = 200;
    while (k > 0) {
        k -= 1;
        const op = ops[k % ops.len];
        try b.print(gpa, "  v{d} = {s}{s} v{d} v{d}\n", .{ tmp, op, ty, 100 + k, acc });
        acc = tmp;
        tmp += 1;
    }
    try b.print(gpa, "  store{s} v6 {d} v{d}\n", .{ ty, 16 * nl + 64, acc });
    try b.print(gpa, "  v2 = add v2 v4\n  cbr lt v2 v1 loop done\ndone:\n  v7 = const 0\n", .{});
    for (0..nr) |r| try b.print(gpa, "  v7 = mul v7 v3\n  v7 = add v7 v{d}\n", .{300 + r});
    try b.print(gpa, "  v7 = add v7 v2\n  ret v7\n", .{});
    return b.toOwnedSlice(gpa);
}

test "vecnative: register pressure (spilled vectors, many accumulators)" {
    var fails: usize = 0;
    const cases = [_]struct { []const u8, i64, h.Fill, []const []const u8, usize }{
        .{ "", 8, .i64, &.{ "add", "sub", "xor", "and", "or" }, 12 },
        .{ "", 8, .i64, &.{ "sub", "add" }, 0 },
        .{ ".f64", 8, .f64s, &.{ "add", "min", "sub", "max", "mul", "div" }, 0 },
        .{ ".f32", 4, .f32s, &.{ "max", "add", "min", "sub", "mul", "div" }, 0 },
        .{ ".f64", 8, .f64, &.{ "add", "sub", "mul" }, 0 },
    };
    for (cases) |cs| for ([_]usize{ 4, 12, 18, 24, 30 }) |nl| {
        const nr: usize = if (cs[4] != 0) @min(nl, 20) else 0;
        const src = try pressureSrc(cs[0], cs[1], nl, nr, cs[3]);
        defer gpa.free(src);
        var sets: std.ArrayList([2]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 40 }) |n| try sets.append(gpa, .{ 1024, n });
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src, 1, slices, cs[2], .{}) catch |e| {
            fails += 1;
            std.debug.print("pressure FAIL {s} ty={s} nl={d} nr={d}\n", .{ @errorName(e), cs[0], nl, nr });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

fn lanesSrc(vt: []const u8, lane: []const u8, nl: usize, op: []const u8, kconst: []const u8, pressure: usize) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    try b.print(gpa, "func lanes 2\ne:\n", .{});
    try b.print(gpa, "  v2 = load.{s} v0 0\n  v3 = load.{s} v0 16\n", .{ vt, vt });
    try b.print(gpa, "  v4 = {s}.{s} v2 v3\n  store.{s} v1 0 v4\n", .{ op, vt, vt });
    try b.print(gpa, "  v5 = {s}.{s} v3 v2\n  store.{s} v1 16 v5\n", .{ op, vt, vt });
    try b.print(gpa, "  v6 = {s}.{s} v2 v2\n  store.{s} v1 32 v6\n", .{ op, vt, vt });
    const ld = if (std.mem.eql(u8, lane, "i64")) "load" else if (std.mem.eql(u8, lane, "f32")) "load.f32" else "load.f64";
    try b.print(gpa, "  v7 = {s} v0 40\n  v8 = splat.{s} v7\n  v9 = {s}.{s} v8 v2\n  store.{s} v1 48 v9\n  store.{s} v1 64 v8\n", .{ ld, vt, op, vt, vt, vt });
    try b.print(gpa, "  v10 = {s}\n  v11 = splat.{s} v10\n  v12 = {s}.{s} v3 v11\n  store.{s} v1 80 v12\n  store.{s} v1 96 v11\n", .{ kconst, vt, op, vt, vt, vt });
    // extracts of every lane of v4 and v12
    const st = if (std.mem.eql(u8, lane, "i64")) "store" else if (std.mem.eql(u8, lane, "f32")) "store.f32" else "store.f64";
    try b.print(gpa, "  v13 = const 0\n", .{});
    for (0..nl) |l| {
        try b.print(gpa, "  v{d} = extract.{s} v4 {d}\n  {s} v1 {d} v{d}\n", .{ 20 + l, vt, l, st, 112 + 8 * l, 20 + l });
        try b.print(gpa, "  v{d} = extract.{s} v12 {d}\n  {s} v1 {d} v{d}\n", .{ 30 + l, vt, l, st, 160 + 8 * l, 30 + l });
        if (std.mem.eql(u8, lane, "i64")) try b.print(gpa, "  v13 = mul v13 v13\n  v13 = add v13 v{d}\n", .{20 + l});
    }
    // pressure: many live vectors, combined pairwise
    for (0..pressure) |k| try b.print(gpa, "  v{d} = load.{s} v0 {d}\n", .{ 100 + k, vt, 48 + 16 * k });
    for (0..pressure) |k| try b.print(gpa, "  v{d} = {s}.{s} v{d} v{d}\n  store.{s} v1 {d} v{d}\n", .{ 200 + k, op, vt, 100 + k, 100 + (k * 7 + 3) % pressure, vt, 256 + 16 * k, 200 + k });
    if (pressure > 0) for (0..nl) |l| try b.print(gpa, "  v{d} = extract.{s} v{d} {d}\n  {s} v1 {d} v{d}\n", .{ 300 + l, vt, 200 + pressure - 1, l, st, 1024 + 8 * l, 300 + l });
    try b.print(gpa, "  ret v13\n", .{});
    return b.toOwnedSlice(gpa);
}

test "vecnative: lane semantics of vector ops, splat and extract" {
    var fails: usize = 0;
    const V = struct { []const u8, []const u8, usize, []const []const u8, []const []const u8, []const h.Fill };
    const vts = [_]V{
        .{ "v2i64", "i64", 2, &.{ "add", "sub", "and", "or", "xor", "mul" }, &.{ "const 0", "const -1", "const 81985529216486895" }, &.{.i64} },
        .{ "v4i32", "i64", 4, &.{ "add", "sub", "mul" }, &.{ "const 0", "const -1", "const 4294967297", "const -2147483649" }, &.{.i64} },
        .{ "v2f64", "f64", 2, &.{ "add", "sub", "mul", "div", "min", "max" }, &.{ "fconst.f64 0", "fconst.f64 -0", "fconst.f64 1.5", "fconst.f64 nan" }, &.{ .f64s, .f64 } },
        .{ "v4f32", "f32", 4, &.{ "add", "sub", "mul", "div", "min", "max" }, &.{ "fconst.f32 0", "fconst.f32 -0", "fconst.f32 1.5", "fconst.f32 nan" }, &.{ .f32s, .f32 } },
    };
    for (vts) |v| for (v[3]) |op| for (v[4]) |kc| for (v[5]) |fill| for ([_]usize{ 0, 20, 34 }) |pr| {
        const src = try lanesSrc(v[0], v[1], v[2], op, kc, pr);
        defer gpa.free(src);
        var sets: [8][2]i64 = undefined;
        for (&sets, 0..) |*s, i| s.* = .{ 1024 + 16 * @as(i64, @intCast(i)) + @as(i64, if (i % 2 == 1) 8 else 0), 8192 };
        var slices: [8][]const i64 = undefined;
        for (&slices, &sets) |*s, *x| s.* = x;
        runSets(src, 3, &slices, fill, .{ .min_vec = 0 }) catch |e| {
            fails += 1;
            std.debug.print("lanes FAIL {s} vt={s} op={s} k={s} fill={s} pressure={d}\n", .{ @errorName(e), v[0], op, kc, @tagName(fill), pr });
            if (fails > 6) return e;
        };
    };
    std.debug.print("lanes: runs {d} native {d}\n", .{ h.run_count, h.native_count });
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: constant trip bounds folded by sccp" {
    var fails: usize = 0;
    const starts = [_]i64{ 0, -3, minI, maxI - 3, maxI, -1, 7 };
    const deltas = [_]i64{ -1, 0, 1, 2, 3, 4, 5, 8, 9, 17 };
    for ([_]bool{ false, true }) |top| for (0..2) |form| for (rels) |rel| for ([_]i64{ 1, 2, 8 }) |step| for (starts) |s| for (deltas) |d| {
        const src0 = try tripSrc(top, false, @intCast(form), rel, step, 3);
        defer gpa.free(src0);
        const ins = try std.fmt.allocPrint(gpa, "e:\n  v1 = const {d}\n  v2 = const {d}\n", .{ s, s +% 3 +% d *% step });
        defer gpa.free(ins);
        const src = try std.mem.replaceOwned(u8, gpa, src0, "e:\n", ins);
        defer gpa.free(src);
        const set = [_]i64{ 1024, 0, 0 };
        const slices = [_][]const i64{&set};
        runSets(src, 1, &slices, .i64, .{ .min_vec = 0 }) catch |e| {
            fails += 1;
            std.debug.print("const FAIL {s} top={} rel={s} step={d} s={d} d={d}\n", .{ @errorName(e), top, rel, step, s, d });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: extra induction variables with odd steps, exit values" {
    var fails: usize = 0;
    for ([_]i64{ -3, 0, maxI, minI, 5, 1 << 62 }) |k| for ([_]bool{ false, true }) |top| {
        const src = if (top) try std.fmt.allocPrint(gpa,
            \\func ivs 3
            \\e:
            \\  v3 = const 0
            \\  v4 = const 8
            \\  v5 = const 1
            \\  v6 = copy v2
            \\  v7 = const {d}
            \\  v12 = copy v2
            \\  br head
            \\head:
            \\  v13 = mul v6 v4
            \\  cbr lt v3 v1 body done
            \\body:
            \\  v8 = mul v3 v4
            \\  v9 = add v0 v8
            \\  v10 = load v9 0
            \\  v10 = add v10 v12
            \\  store v9 0 v10
            \\  v6 = add v6 v7
            \\  v3 = add v3 v5
            \\  br head
            \\done:
            \\  v11 = mul v6 v4
            \\  v11 = add v11 v3
            \\  v11 = xor v11 v13
            \\  v11 = add v11 v12
            \\  ret v11
        , .{k}) else try std.fmt.allocPrint(gpa,
            \\func ivs 3
            \\e:
            \\  v3 = const 0
            \\  v4 = const 8
            \\  v5 = const 1
            \\  v6 = copy v2
            \\  v7 = const {d}
            \\  v12 = copy v2
            \\  br loop
            \\loop:
            \\  v13 = mul v6 v4
            \\  v8 = mul v3 v4
            \\  v9 = add v0 v8
            \\  v10 = load v9 0
            \\  v10 = add v10 v12
            \\  store v9 0 v10
            \\  v6 = sub v6 v7
            \\  v3 = add v3 v5
            \\  cbr lt v3 v1 loop done
            \\done:
            \\  v11 = mul v6 v4
            \\  v11 = add v11 v3
            \\  v11 = xor v11 v13
            \\  v11 = add v11 v12
            \\  ret v11
        , .{k});
        defer gpa.free(src);
        var sets: std.ArrayList([3]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 1, 2, 3, 4, 5, 8, 9, 17, -1 }) |n| for ([_]i64{ 0, -7, maxI }) |init| try sets.append(gpa, .{ 1024, n, init });
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src, 1, slices, .i64, .{}) catch |e| {
            fails += 1;
            std.debug.print("ivs FAIL {s} k={d} top={}\n", .{ @errorName(e), k, top });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: undefined vregs (read 0) as splat source, store value, reduction init and bound" {
    var fails: usize = 0;
    const srcs = [_][]const u8{
        \\func u1 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  br loop
        \\loop:
        \\  v5 = mul v2 v3
        \\  v6 = add v0 v5
        \\  v7 = load v6 0
        \\  v8 = add v7 v50
        \\  v9 = xor v8 v51
        \\  store v6 0 v9
        \\  v52 = add v52 v7
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  v10 = add v52 v2
        \\  ret v10
        ,
        \\func u2 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  br loop
        \\loop:
        \\  v5 = mul v2 v3
        \\  v6 = add v0 v5
        \\  store v6 0 v50
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  ret v2
        ,
        \\func u3 2
        \\e:
        \\  v3 = const 8
        \\  v4 = const 1
        \\  br loop
        \\loop:
        \\  v5 = mul v2 v3
        \\  v6 = add v0 v5
        \\  v7 = load v6 0
        \\  v52 = xor v52 v7
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  v10 = add v52 v2
        \\  ret v10
        ,
        \\func u4 2
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  v20 = add v1 v4
        \\  br loop
        \\loop:
        \\  v5 = mul v2 v3
        \\  v6 = add v0 v5
        \\  v7 = load.f64 v6 0
        \\  v8 = cvt.f64.i64 v50
        \\  v9 = max.f64 v7 v8
        \\  store.f64 v6 0 v9
        \\  v2 = add v2 v4
        \\  v21 = add v2 v51
        \\  cbr lt v21 v1 loop done
        \\done:
        \\  ret v2
    };
    for (srcs) |src| {
        var sets: std.ArrayList([2]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 1, 2, 3, 4, 5, 8, 9, 17 }) |n| try sets.append(gpa, .{ 1024, n });
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src, 1, slices, .i64, .{}) catch |e| {
            fails += 1;
            std.debug.print("undef FAIL {s}\n", .{@errorName(e)});
        };
    }
    try t.expectEqual(@as(usize, 0), fails);
}

/// Random vectorizable-ish loops.
fn randomLoop(r: std.Random, b: *std.ArrayList(u8)) !struct { fill: h.Fill, top: bool } {
    const kind = r.intRangeLessThan(u8, 0, 3); // 0 i64, 1 f64, 2 f32
    const e: i64 = if (kind == 2) 4 else 8;
    const ty = switch (kind) {
        0 => "",
        1 => ".f64",
        else => ".f32",
    };
    const steps = [_]i64{ 1, 2, 4, 8 };
    var step = steps[r.intRangeLessThan(usize, 0, if (e == 4) 3 else 4)];
    _ = &step;
    const top = r.boolean();
    const rel = rels[r.intRangeLessThan(usize, 0, rels.len)];
    const form = r.intRangeLessThan(u8, 0, 3);
    const offs = [_]i64{ 0, 0, step, -step, 3, 2 * step };
    const off = offs[r.intRangeLessThan(usize, 0, offs.len)];
    const use_piv = r.boolean();
    // v0 A, v1 B, v2 n, v3 s, v4 k(int uniform)
    try b.print(gpa, "func rnd 5\ne:\n  v5 = copy v3\n  v6 = const {d}\n  v7 = const {d}\n  v8 = const {d}\n  v9 = copy v1\n  v10 = const {d}\n", .{ step, @divExact(e, step), off, e });
    const nred: usize = if (kind == 0) r.intRangeAtMost(usize, 0, 3) else 0;
    const red_ops = [_][]const u8{ "add", "sub", "and", "or", "xor" };
    var rop: [3][]const u8 = undefined;
    for (0..nred) |j| {
        rop[j] = red_ops[r.intRangeLessThan(usize, 0, red_ops.len)];
        try b.print(gpa, "  v{d} = add v4 v{d}\n", .{ 20 + j, 6 + j });
    }
    if (kind != 0) try b.print(gpa, "  v15 = cvt{s}.i64 v4\n", .{ty});
    var tb: [128]u8 = undefined;
    if (top) {
        try b.print(gpa, "  br head\nhead:\n  v11 = add v5 v8\n  {s}\nbody:\n", .{try testText(&tb, form, rel, "v11", "v2", "body", "done")});
    } else {
        try b.print(gpa, "  br loop\nloop:\n", .{});
    }
    try b.print(gpa, "  v12 = sub v5 v3\n  v12 = mul v12 v7\n  v13 = add v0 v12\n  v14 = add v9 v10\n", .{});
    // data values: v100.. ; uniform available: v15 (float) or v4 (int)
    var nd: usize = 0;
    const nops = r.intRangeAtMost(usize, 2, 10);
    const iops = [_][]const u8{ "add", "sub", "and", "or", "xor" };
    const fops = [_][]const u8{ "add", "sub", "mul", "div", "min", "max" };
    const uni = if (kind == 0) "v4" else "v15";
    var buf2: [16]u8 = undefined;
    for (0..nops) |_| {
        const what = r.intRangeLessThan(u8, 0, 10);
        const base = if (use_piv and r.boolean()) "v9" else "v13";
        const disp = if (r.intRangeLessThan(u8, 0, 4) == 0) r.intRangeAtMost(i64, -40, 40) else e * r.intRangeAtMost(i64, -4, 4);
        if (what < 3 or nd == 0) {
            try b.print(gpa, "  v{d} = load{s} {s} {d}\n", .{ 100 + nd, ty, base, disp });
            nd += 1;
        } else if (what < 7) {
            const op = if (kind == 0) iops[r.intRangeLessThan(usize, 0, iops.len)] else fops[r.intRangeLessThan(usize, 0, fops.len)];
            const x = try std.fmt.bufPrint(&buf2, "v{d}", .{100 + r.intRangeLessThan(usize, 0, nd)});
            var xb: [16]u8 = undefined;
            const xs = try std.fmt.bufPrint(&xb, "{s}", .{x});
            const y = if (r.intRangeLessThan(u8, 0, 4) == 0) uni else try std.fmt.bufPrint(&buf2, "v{d}", .{100 + r.intRangeLessThan(usize, 0, nd)});
            if (r.boolean())
                try b.print(gpa, "  v{d} = {s}{s} {s} {s}\n", .{ 100 + nd, op, ty, xs, y })
            else
                try b.print(gpa, "  v{d} = {s}{s} {s} {s}\n", .{ 100 + nd, op, ty, y, xs });
            nd += 1;
        } else {
            const src = if (r.intRangeLessThan(u8, 0, 5) == 0) uni else try std.fmt.bufPrint(&buf2, "v{d}", .{100 + r.intRangeLessThan(usize, 0, nd)});
            try b.print(gpa, "  store{s} {s} {d} {s}\n", .{ ty, base, disp, src });
        }
    }
    for (0..nred) |j| try b.print(gpa, "  v{d} = {s} v{d} v{d}\n", .{ 20 + j, rop[j], 20 + j, 100 + r.intRangeLessThan(usize, 0, nd) });
    try b.print(gpa, "  v9 = add v9 v10\n  v5 = add v5 v6\n", .{});
    if (top) {
        try b.print(gpa, "  br head\n", .{});
    } else {
        try b.print(gpa, "  v11 = add v5 v8\n  {s}\n", .{try testText(&tb, form, rel, "v11", "v2", "loop", "done")});
    }
    try b.print(gpa, "done:\n  v30 = sub v9 v1\n  v30 = xor v30 v5\n", .{});
    for (0..nred) |j| try b.print(gpa, "  v30 = mul v30 v7\n  v30 = add v30 v{d}\n", .{20 + j});
    if (!top) try b.print(gpa, "  v30 = add v30 v11\n  v30 = add v30 v12\n", .{});
    try b.print(gpa, "  ret v30\n", .{});
    return .{ .fill = switch (kind) {
        0 => .i64,
        1 => if (r.boolean()) .f64 else .f64s,
        else => if (r.boolean()) .f32 else .f32s,
    }, .top = top };
}

test "vecnative: random loops" {
    try randomLoops(false);
}

test "vecnative: random loops with random profiles" {
    try randomLoops(true);
}

test "vecnative: random loops optimized twice" {
    twice = true;
    defer twice = false;
    try randomLoops(false);
}
var twice = false;

fn randomLoops(with_profile: bool) !void {
    const count: usize = if (std.c.getenv("VEC_FUZZ")) |s| std.fmt.parseInt(usize, std.mem.span(s), 10) catch 1500 else 1500;
    var prng = std.Random.DefaultPrng.init(if (std.c.getenv("VEC_SEED")) |s| std.fmt.parseInt(u64, std.mem.span(s), 10) catch 11 else 11);
    const r = prng.random();
    var fails: usize = 0;
    const vec0 = h.vec_count;
    for (0..count) |_| {
        var b: std.ArrayList(u8) = .empty;
        defer b.deinit(gpa);
        const info = try randomLoop(r, &b);
        var sets: std.ArrayList([5]i64) = .empty;
        defer sets.deinit(gpa);
        for (0..24) |_| {
            const starts = [_]i64{ 0, 0, 5, -3, minI, maxI - 9, -1 };
            const s = starts[r.intRangeLessThan(usize, 0, starts.len)];
            const n = if (r.intRangeLessThan(u8, 0, 6) == 0) starts[r.intRangeLessThan(usize, 0, starts.len)] else s +% r.intRangeAtMost(i64, -3, 40);
            const boff = if (r.boolean()) r.intRangeAtMost(i64, -48, 48) else 8192;
            try sets.append(gpa, .{ 4096, 4096 + boff, n, s, r.intRangeAtMost(i64, -5, 5) });
        }
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        var pb: [16]u64 = undefined;
        var pt: [16]u64 = undefined;
        for (&pb, &pt) |*x, *y| {
            const big = [_]u64{ 0, 1, 7, 1000, 1 << 40, std.math.maxInt(u64), std.math.maxInt(u64) / 3 };
            x.* = if (r.boolean()) big[r.intRangeLessThan(usize, 0, big.len)] else r.intRangeAtMost(u64, 0, 100000);
            y.* = if (r.boolean()) big[r.intRangeLessThan(usize, 0, big.len)] else r.intRangeAtMost(u64, 0, 100000);
        }
        const prof: @import("core/opt/opt.zig").Profile = .{ .blocks = &pb, .taken = &pt };
        h.check(b.items, 3, slices, info.fill, .{ .min_vec = 0, .profile = if (with_profile) &prof else null, .twice = twice }) catch |e| {
            fails += 1;
            std.debug.print("random FAIL {s}\n{s}\n", .{ @errorName(e), b.items });
            if (fails > 3) return e;
        };
    }
    std.debug.print("random: {d} programs, {d} vectorized loops\n", .{ count, h.vec_count - vec0 });
    try t.expectEqual(@as(usize, 0), fails);
}

test "vecnative: dump o2" {
    const src =
        \\func add_f64 4
        \\e:
        \\  v4 = const 0
        \\  v5 = const 8
        \\  v6 = const 1
        \\  cbr ge v4 v3 done loop
        \\loop:
        \\  v7 = mul v4 v5
        \\  v8 = add v0 v7
        \\  v9 = add v1 v7
        \\  v10 = add v2 v7
        \\  v11 = load.f64 v8 0
        \\  v12 = load.f64 v9 0
        \\  v13 = add.f64 v11 v12
        \\  store.f64 v10 0 v13
        \\  v4 = add v4 v6
        \\  cbr lt v4 v3 loop done
        \\done:
        \\  ret v4
    ;
    const opt = @import("core/opt/opt.zig");
    const ir = @import("core/ir.zig");
    var f = try ir.parse(gpa, src);
    defer f.deinit();
    _ = try opt.optimize(gpa, &f, .{ .level = .o2, .vector = opt.VectorTarget.host });
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try f.print(gpa, &out);
    std.debug.print("{s}\n", .{out.items});
}

test "vecnative: vectors and floats live across calls" {
    if (!h.can_run_pub) return;
    // Callee: clobbers many vector registers with values derived from its buffer.
    var cs: std.ArrayList(u8) = .empty;
    defer cs.deinit(gpa);
    try cs.print(gpa, "func clob 2\ne:\n", .{});
    for (0..30) |k| try cs.print(gpa, "  v{d} = load.v2i64 v0 {d}\n", .{ 10 + k, 16 * k });
    for (0..30) |k| try cs.print(gpa, "  v{d} = xor.v2i64 v{d} v{d}\n  store.v2i64 v0 {d} v{d}\n", .{ 50 + k, 10 + k, 10 + (k + 1) % 30, 16 * k, 50 + k });
    try cs.print(gpa, "  v2 = fconst.f64 7.5\n  v3 = cvt.i64.f64 v2\n  ret v3\n", .{});
    const callee = try h.compileNative(cs.items);
    defer callee.release();
    const addr = @intFromPtr(callee.mem.ptr);
    const src = try std.fmt.allocPrint(gpa,
        \\func vc 2
        \\e:
        \\  v2 = load.v2i64 v0 0
        \\  v3 = load.v4f32 v0 16
        \\  v4 = load.v2f64 v0 32
        \\  v7 = load.f64 v0 48
        \\  v8 = load.f32 v0 56
        \\  v20 = const 4096
        \\  v21 = add v1 v20
        \\  v5 = call {d} v21 v21
        \\  v6 = add.v2i64 v2 v2
        \\  store.v2i64 v1 0 v6
        \\  store.v4f32 v1 16 v3
        \\  store.v2f64 v1 32 v4
        \\  store.f64 v1 48 v7
        \\  store.f32 v1 56 v8
        \\  v9 = call {d} v21 v21
        \\  v10 = add.v4f32 v3 v3
        \\  store.v4f32 v1 64 v10
        \\  v11 = extract.v2f64 v4 1
        \\  store.f64 v1 80 v11
        \\  v12 = extract.v2i64 v2 1
        \\  v13 = add v12 v9
        \\  ret v13
    , .{ addr, addr });
    defer gpa.free(src);
    const set = [_]i64{ 1024, 8192 };
    const slices = [_][]const i64{&set};
    try runSets(src, 3, &slices, .i64, .{ .min_vec = 0 });
}

test "vecnative: vector loads/stores with large and odd displacements" {
    var fails: usize = 0;
    const tys = [_]struct { []const u8, i64, h.Fill }{ .{ "", 8, .i64 }, .{ ".f64", 8, .f64 }, .{ ".f32", 4, .f32 } };
    for (tys) |ty| for ([_]i64{ 65520, 65536, 65528, 4096 * 16 + 8, -257, -256, -4096, -65536, -70004, 255, 256, 4080, 4088, 4100 }) |disp| {
        const src = try std.fmt.allocPrint(gpa,
            \\func disp 3
            \\e:
            \\  v4 = const 0
            \\  v5 = const {d}
            \\  v6 = const 1
            \\  cbr ge v4 v2 done loop
            \\loop:
            \\  v7 = mul v4 v5
            \\  v8 = add v0 v7
            \\  v9 = add v1 v7
            \\  v10 = load{s} v9 {d}
            \\  v11 = load{s} v8 0
            \\  v12 = add{s} v10 v11
            \\  store{s} v9 {d} v12
            \\  v4 = add v4 v6
            \\  cbr lt v4 v2 loop done
            \\done:
            \\  ret v4
        , .{ ty[1], ty[0], disp, ty[0], ty[0], ty[0], disp });
        defer gpa.free(src);
        var sets: std.ArrayList([3]i64) = .empty;
        defer sets.deinit(gpa);
        for ([_]i64{ 0, 1, 2, 3, 4, 5, 8, 9, 17 }) |n| try sets.append(gpa, .{ 8, 8192 - disp, n }); // every access stays inside the buffer
        const slices = try gpa.alloc([]const i64, sets.items.len);
        defer gpa.free(slices);
        for (sets.items, 0..) |*s, i| slices[i] = s;
        runSets(src, 3, slices, ty[2], .{}) catch |e| {
            fails += 1;
            std.debug.print("disp FAIL {s} ty={s} disp={d}\n", .{ @errorName(e), ty[0], disp });
            if (fails > 4) return e;
        };
    };
    try t.expectEqual(@as(usize, 0), fails);
}

fn optOnce(alloc: std.mem.Allocator, src: []const u8) !void {
    const opt = @import("core/opt/opt.zig");
    const ir = @import("core/ir.zig");
    var f = try ir.parse(std.heap.page_allocator, src);
    defer f.deinit();
    _ = try opt.optimize(alloc, &f, .{ .level = .o2, .vector = opt.VectorTarget.host, .verify_each = false });
}

test "vecnative: allocation failures during vectorization do not leak" {
    const src =
        \\func esc 3
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  v5 = const 0
        \\  br loop
        \\loop:
        \\  v6 = mul v2 v3
        \\  v7 = add v0 v6
        \\  v13 = add v12 v6
        \\  v8 = load v7 0
        \\  store v13 0 v8
        \\  v5 = xor v5 v8
        \\  v9 = mul v1 v3
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  v10 = add v5 v2
        \\  v11 = add v10 v9
        \\  v12 = add v11 v6
        \\  ret v12
    ;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, optOnce, .{src});
}

test "vecnative: allocation failures inside vectorize.run do not leak" {
    const opt = @import("core/opt/opt.zig");
    const ir = @import("core/ir.zig");
    const srcs = [_][]const u8{
        \\func esc 3
        \\e:
        \\  v2 = const 0
        \\  v3 = const 8
        \\  v4 = const 1
        \\  v5 = const 0
        \\  br loop
        \\loop:
        \\  v6 = mul v2 v3
        \\  v7 = add v0 v6
        \\  v13 = add v2 v6
        \\  v8 = load v7 0
        \\  store v13 0 v8
        \\  v5 = xor v5 v8
        \\  v9 = mul v1 v3
        \\  v2 = add v2 v4
        \\  cbr lt v2 v1 loop done
        \\done:
        \\  v10 = add v5 v2
        \\  v11 = add v10 v9
        \\  v12 = add v11 v6
        \\  ret v12
        ,
        \\func seq 3
        \\e:
        \\  v3 = const 0
        \\  v4 = const 8
        \\  v5 = const 1
        \\  v6 = const 0
        \\  v14 = add v2 v2
        \\  br l1
        \\l1:
        \\  v7 = mul v3 v4
        \\  v8 = add v0 v7
        \\  v9 = load v8 0
        \\  v6 = add v6 v9
        \\  v3 = add v3 v5
        \\  cbr lt v3 v2 l1 l2
        \\l2:
        \\  v10 = mul v3 v4
        \\  v11 = add v1 v10
        \\  v12 = load v11 0
        \\  v13 = xor v12 v6
        \\  store v11 0 v13
        \\  v3 = add v3 v5
        \\  cbr lt v3 v14 l2 done
        \\done:
        \\  v15 = sub v6 v3
        \\  ret v15
    };
    for (srcs) |src| {
        var i: usize = 0;
        while (i < 2000) : (i += 1) {
            var f = try ir.parse(std.testing.allocator, src);
            defer f.deinit();
            const options: opt.Options = .{ .vector = opt.VectorTarget.host };
            var stats: opt.Stats = .{};
            var c = try opt.Ctx.init(std.testing.allocator, &f, &options, &stats);
            defer c.deinit();
            try @import("core/opt/ssa.zig").build(&c);
            var fa = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = i });
            const saved = c.gpa;
            c.gpa = fa.allocator();
            const r = @import("core/opt/vectorize.zig").run(&c);
            c.gpa = saved;
            if (r) |_| {
                if (!fa.has_induced_failure) break;
            } else |e| try std.testing.expectEqual(error.OutOfMemory, e);
        }
        std.debug.print("oom: {d} fail points\n", .{i});
    }
}
