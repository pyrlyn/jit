//! Encodings of every form of src/x86_64/vec_table.zig (generated from LLVM, see
//! docs/llvm-tables.md), one or more cases per form, data in src/testdata/vec_cases.txt
//! (written by scripts/gen_vec_tests.py; do not edit by hand).
//!
//! Each case line: `MODE MNEMONIC OPS EXTRA EXPECT SOURCE`
//!
//!   MODE    x64 | x86
//!   OPS     comma-separated, `-` for none: xmm1 ymm2 zmm3 mm1 k1 tmm1 st2 seg4 cr3 dr2,
//!           r8:dl r16:dx r32:edx r64:rdx (GP register), i5 (immediate), mN ([rcx+128], N bytes,
//!           0 = unsized), bExC (broadcast: element E bytes, C times), vxN/vyN/vzN (VSIB
//!           [rcx+xmm5/ymm5/zmm5*4+128], N-byte index), ma ([rcx] address register)
//!   EXTRA   `-` or a combination of k7 z rn sae evex
//!   EXPECT  hex bytes, or `ERR` (no matching form)
//!   SOURCE  asmjit: the bytes the AsmJit-derived table produced (frozen before the switch to
//!           LLVM); llvm: checked against llvm-mc 23.1.4; removed: a form the AsmJit table had
//!           and LLVM does not define (see docs/llvm-tables.md)
const std = @import("std");
const opts = @import("pyrlyn_jit_options");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const vec = @import("x86_64/vec.zig");
const gp = @import("x86_64/gp.zig");

pub const Case = struct {
    x86: bool,
    mn: []const u8,
    ops: [8]vec.Op = undefined,
    n: usize = 0,
    e: vec.Extra = .{},
    expect: ?[]const u8,
    source: []const u8,
};

fn gpReg(name: []const u8) !gp.Reg {
    const names = [_][]const u8{ "ax", "cx", "dx", "bx", "sp", "bp", "si", "di" };
    const l8 = [_][]const u8{ "al", "cl", "dl", "bl" };
    for (l8, 0..) |s, i| if (std.mem.eql(u8, s, name)) return gp.r8(@fromBackingInt(@intCast(i)));
    const core = if (name.len == 3 and (name[0] == 'e' or name[0] == 'r')) name[1..] else name;
    for (names, 0..) |s, i| if (std.mem.eql(u8, s, core)) {
        const g: @import("x86_64/assembler.zig").Gp = @fromBackingInt(@intCast(i));
        return if (name.len == 2) gp.r16(g) else if (name[0] == 'e') gp.r32(g) else gp.r64(g);
    };
    return error.BadCase;
}

fn parseOp(t: []const u8) !vec.Op {
    const num = struct {
        fn f(s: []const u8) !u16 {
            return std.fmt.parseInt(u16, s, 10) catch error.BadCase;
        }
    }.f;
    const regs = [_]struct { []const u8, vec.Cls }{
        .{ "xmm", .xmm }, .{ "ymm", .ymm }, .{ "zmm", .zmm }, .{ "mm", .mm }, .{ "tmm", .tmm },
        .{ "k", .k },     .{ "st", .st },   .{ "seg", .seg }, .{ "cr", .cr }, .{ "dr", .dr },
    };
    if (std.mem.indexOfScalar(u8, t, ':')) |c| return .{ .r = try gpReg(t[c + 1 ..]) };
    if (std.mem.eql(u8, t, "ma")) return .{ .m = .{ .base = .rcx } };
    if (t[0] == 'i') return .{ .imm = try std.fmt.parseInt(i64, t[1..], 10) };
    if (t[0] == 'm' and t.len > 1 and std.ascii.isDigit(t[1])) return .{ .m = .{ .base = .rcx, .disp = 128, .size = try num(t[1..]) } };
    if (t[0] == 'b' and t.len > 1 and std.ascii.isDigit(t[1])) {
        const xi = std.mem.indexOfScalar(u8, t, 'x') orelse return error.BadCase;
        return .{ .m = .{ .base = .rcx, .disp = 128, .size = try num(t[1..xi]), .bcst = @intCast(try num(t[xi + 1 ..])) } };
    }
    if (t[0] == 'v' and t.len > 2 and std.ascii.isDigit(t[2])) {
        const v: vec.V = switch (t[1]) {
            'x' => vec.xmm(5),
            'y' => vec.ymm(5),
            'z' => vec.zmm(5),
            else => return error.BadCase,
        };
        return .{ .m = vec.vsib(.rcx, v, .x4, 128, try num(t[2..])) };
    }
    for (regs) |r| if (std.mem.startsWith(u8, t, r[0]) and t.len > r[0].len and std.ascii.isDigit(t[r[0].len])) {
        return .{ .v = .{ .cls = r[1], .id = @intCast(try num(t[r[0].len..])) } };
    };
    return error.BadCase;
}

pub fn parse(a: std.mem.Allocator, line: []const u8) !Case {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    const mode = it.next() orelse return error.BadCase;
    var c = Case{ .x86 = std.mem.eql(u8, mode, "x86"), .mn = it.next() orelse return error.BadCase, .expect = null, .source = "" };
    const ops = it.next() orelse return error.BadCase;
    if (!std.mem.eql(u8, ops, "-")) {
        var oi = std.mem.splitScalar(u8, ops, ',');
        while (oi.next()) |o| {
            c.ops[c.n] = try parseOp(o);
            c.n += 1;
        }
    }
    const ext = it.next() orelse return error.BadCase;
    if (std.mem.indexOf(u8, ext, "k7") != null) c.e.k = 7;
    if (std.mem.indexOf(u8, ext, "z") != null) c.e.z = true;
    if (std.mem.indexOf(u8, ext, "rn") != null) c.e.rc = .rn;
    if (std.mem.indexOf(u8, ext, "sae") != null) c.e.sae = true;
    if (std.mem.indexOf(u8, ext, "evex") != null) c.e.evex = true;
    const ex = it.next() orelse return error.BadCase;
    if (!std.mem.eql(u8, ex, "ERR")) {
        const b = try a.alloc(u8, ex.len / 2);
        _ = std.fmt.hexToBytes(b, ex) catch return error.BadCase;
        c.expect = b;
    }
    c.source = it.next() orelse "";
    return c;
}

/// Forms of every mnemonic of a table module (`pub const table = struct { ... }`), by name.
pub fn Lookup(comptime T: type) type {
    return struct {
        const Entry = struct { []const u8, []const vec.Form };
        const entries = blk: {
            @setEvalBranchQuota(2_000_000);
            const decls = @typeInfo(T.table).@"struct".decl_names;
            var e: [decls.len]Entry = undefined;
            for (decls, 0..) |d, i| e[i] = .{ d, &@field(T.table, d) };
            const final = e;
            break :blk final;
        };
        const map = std.StaticStringMap([]const vec.Form).initComptime(entries);
        pub fn get(name: []const u8) ?[]const vec.Form {
            return map.get(name);
        }
    };
}

/// Encodes a case with `forms`; null if no form matches.
pub fn encode(cb: *CodeBuffer, forms: []const vec.Form, c: Case) !?[]const u8 {
    cb.reset();
    gp.mode32 = c.x86;
    defer gp.mode32 = false;
    vec.emitOps(cb, forms, c.ops[0..c.n], c.e) catch |err| switch (err) {
        error.NoMatchingForm, error.InvalidOperand => return null,
        else => return err,
    };
    return cb.code();
}

const cases_txt = @embedFile("testdata/vec_cases.txt");

test "x86_64 vec_table: every form (frozen AsmJit-table encodings + llvm-mc)" {
    if (!opts.x86_64) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const L = Lookup(vec.tbl);
    var fails: u32 = 0;
    var n: u32 = 0;
    var it = std.mem.splitScalar(u8, cases_txt, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (line.len == 0 or line[0] == '#') continue;
        const c = try parse(a, line);
        n += 1;
        const forms = L.get(c.mn) orelse {
            std.debug.print("vec case: unknown mnemonic: {s}\n", .{line});
            fails += 1;
            continue;
        };
        const got = try encode(&cb, forms, c);
        const ok = if (c.expect) |want| (if (got) |g| std.mem.eql(u8, g, want) else false) else got == null;
        if (!ok) {
            fails += 1;
            if (got) |g| std.debug.print("vec case: {s}: got {X}\n", .{ line, g }) else std.debug.print("vec case: {s}: no matching form\n", .{line});
        }
    }
    try std.testing.expect(n > 0);
    try std.testing.expectEqual(@as(u32, 0), fails);
}
