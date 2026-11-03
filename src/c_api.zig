//! Stable C ABI (see include/pyrlyn_jit.h). Every entry point validates its arguments and returns a
//! `pjit_status`; no Zig panics or UB are reachable through bad input from C.
const std = @import("std");
const builtin = @import("builtin");
const cbm = @import("core/code_buffer.zig");
const rt = @import("core/jit_runtime.zig");
const pool = @import("core/jit_pool.zig");
const gp = @import("x86_64/gp.zig");
const x86 = @import("x86_64/assembler.zig");
const a64 = @import("aarch64/a64.zig");

pub const version: u32 = 0x00_01_00; // 0.1.0, MAJOR<<16 | MINOR<<8 | PATCH

const Status = enum(c_int) {
    ok = 0,
    out_of_memory,
    invalid_argument,
    unknown_mnemonic,
    no_matching_form,
    invalid_label,
    label_already_bound,
    unbound_label,
    displacement_out_of_range,
    invalid_fixup,
    code_too_large,
    map_failed,
    protect_failed,
    other,
};

fn status(e: anyerror) c_int {
    const s: Status = switch (e) {
        error.OutOfMemory => .out_of_memory,
        error.InvalidOperand, error.InvalidImmediate, error.OutOfBounds => .invalid_argument,
        error.NoMatchingForm => .no_matching_form,
        error.InvalidLabel, error.TooManyLabels => .invalid_label,
        error.LabelAlreadyBound => .label_already_bound,
        error.UnboundLabel => .unbound_label,
        error.DisplacementOutOfRange => .displacement_out_of_range,
        error.InvalidFixup => .invalid_fixup,
        error.CodeTooLarge => .code_too_large,
        error.MapFailed => .map_failed,
        error.ProtectFailed => .protect_failed,
        else => .other,
    };
    return @backingInt(s);
}
const OK: c_int = 0;
const EINVAL: c_int = @backingInt(Status.invalid_argument);

const gpa = std.heap.c_allocator;

const Buffer = struct { cb: cbm.CodeBuffer };

export fn pjit_version() u32 {
    return version;
}

export fn pjit_status_str(s: c_int) [*:0]const u8 {
    const e = std.enums.fromInt(Status, s) orelse return "unknown status";
    return switch (e) {
        .ok => "ok",
        .out_of_memory => "out of memory",
        .invalid_argument => "invalid argument",
        .unknown_mnemonic => "unknown mnemonic",
        .no_matching_form => "no instruction form matches the operands",
        .invalid_label => "invalid label",
        .label_already_bound => "label already bound",
        .unbound_label => "label referenced but never bound",
        .displacement_out_of_range => "branch displacement out of range",
        .invalid_fixup => "fixup outside the emitted code",
        .code_too_large => "code too large",
        .map_failed => "mapping executable memory failed",
        .protect_failed => "changing memory protection failed",
        .other => "internal error",
    };
}

// ---- buffer / labels ------------------------------------------------------------------------

export fn pjit_buffer_new(out: ?*?*Buffer) c_int {
    const o = out orelse return EINVAL;
    const b = gpa.create(Buffer) catch return status(error.OutOfMemory);
    b.* = .{ .cb = cbm.CodeBuffer.init(gpa) };
    o.* = b;
    return OK;
}

export fn pjit_buffer_free(b: ?*Buffer) void {
    const p = b orelse return;
    p.cb.deinit();
    gpa.destroy(p);
}

export fn pjit_buffer_reset(b: ?*Buffer) void {
    if (b) |p| p.cb.reset();
}

export fn pjit_buffer_code(b: ?*const Buffer, len: ?*usize) ?[*]const u8 {
    const p = b orelse return null;
    if (len) |l| l.* = p.cb.code().len;
    return p.cb.code().ptr;
}

export fn pjit_buffer_emit_bytes(b: ?*Buffer, bytes: ?[*]const u8, n: usize) c_int {
    const p = b orelse return EINVAL;
    if (n == 0) return OK;
    const src = bytes orelse return EINVAL;
    p.cb.emitBytes(src[0..n]) catch |e| return status(e);
    return OK;
}

export fn pjit_label_new(b: ?*Buffer, out: ?*u32) c_int {
    const p = b orelse return EINVAL;
    const o = out orelse return EINVAL;
    const l = p.cb.newLabel() catch |e| return status(e);
    o.* = l.id;
    return OK;
}

export fn pjit_label_bind(b: ?*Buffer, label: u32) c_int {
    const p = b orelse return EINVAL;
    p.cb.bind(.{ .id = label }) catch |e| return status(e);
    return OK;
}

export fn pjit_finalize(b: ?*Buffer) c_int {
    const p = b orelse return EINVAL;
    p.cb.finalize() catch |e| return status(e);
    return OK;
}

// ---- mnemonic lookup ------------------------------------------------------------------------

fn FormMap(comptime T: type, comptime F: type) type {
    @setEvalBranchQuota(2_000_000);
    const decls = @typeInfo(T).@"struct".decl_names;
    var kvs: [decls.len]struct { []const u8, []const F } = undefined;
    for (decls, 0..) |d, i| kvs[i] = .{ d, &@field(T, d) };
    const final = kvs;
    return struct {
        const map = std.StaticStringMap([]const F).initComptime(final);
    };
}
const a64_map = std.StaticStringMap(a64.Mnemonic).initComptime(blk: {
    @setEvalBranchQuota(2_000_000);
    const fs = @typeInfo(a64.Mnemonic).@"enum".field_names;
    var kvs: [fs.len]struct { []const u8, a64.Mnemonic } = undefined;
    for (fs, 0..) |f, i| kvs[i] = .{ f, @field(a64.Mnemonic, f) };
    break :blk kvs;
});
const gp_map = FormMap(gp.tbl.table, gp.Form).map;

fn lookup(comptime M: anytype, name: ?[*:0]const u8) ?@TypeOf(M.get("").?) {
    const n = name orelse return null;
    return M.get(std.mem.span(n));
}

// ---- AArch64 --------------------------------------------------------------------------------

pub const A64Op = extern struct {
    kind: u8,
    reg: u8,
    sub: u8,
    arr: u8,
    et: u8,
    idx: i8,
    index: u8,
    index64: u8,
    ext_op: u8,
    ext_n: u8,
    has_ext: u8,
    _pad: [5]u8 = @splat(0),
    imm: i64,
    fimm: f64,
};

fn en(comptime E: type, v: anytype) ?E {
    return std.enums.fromInt(E, v);
}

fn a64Op(o: A64Op) ?a64.Op {
    return switch (o.kind) {
        0 => if (o.reg > 32) null else .{ .r = .{ .id = @intCast(o.reg), .x = o.sub != 0 } },
        1 => if (o.reg > 31) null else .{ .v = .{
            .id = @intCast(o.reg),
            .kind = en(a64.VKind, o.sub) orelse return null,
            .arr = en(a64.Arr, o.arr) orelse return null,
            .et = en(a64.Et, o.et) orelse return null,
            .idx = if (o.idx < 0) null else if (o.idx > 15) return null else @intCast(o.idx),
        } },
        2 => .{ .imm = o.imm },
        3 => .{ .fimm = o.fimm },
        4 => .{ .shift = .{ .op = en(a64.ShiftOp, o.sub) orelse return null, .n = std.math.cast(u8, o.imm) orelse return null } },
        5 => if (o.reg > 32 or (o.index != 0xFF and o.index > 31)) null else .{ .mem = .{
            .base = .{ .id = @intCast(o.reg), .x = true },
            .index = if (o.index == 0xFF) null else .{ .id = @intCast(o.index), .x = o.index64 != 0 },
            .ext = if (o.has_ext == 0) null else .{ .op = en(a64.ShiftOp, o.ext_op) orelse return null, .n = o.ext_n },
            .off = o.imm,
            .mode = en(a64.MemMode, o.sub) orelse return null,
        } },
        6 => .{ .cond = std.math.cast(u4, o.imm) orelse return null },
        7 => .{ .rel = o.imm },
        else => null,
    };
}

export fn pjit_a64_emit(b: ?*Buffer, mnemonic: ?[*:0]const u8, ops: ?[*]const A64Op, n: usize) c_int {
    const p = b orelse return EINVAL;
    const mn = lookup(a64_map, mnemonic) orelse return @backingInt(Status.unknown_mnemonic);
    if (n > 8 or (n > 0 and ops == null)) return EINVAL;
    var conv: [8]a64.Op = undefined;
    for (0..n) |i| conv[i] = a64Op(ops.?[i]) orelse return EINVAL;
    const word = a64.encodeMn(mn, conv[0..n]) orelse return status(error.NoMatchingForm);
    p.cb.ensure(4) catch |e| return status(e);
    p.cb.put32(word);
    return OK;
}

/// `b label` (cond < 0) or `b.<cond> label` (cond 0..15).
export fn pjit_a64_branch(b: ?*Buffer, cond: c_int, label: u32) c_int {
    const p = b orelse return EINVAL;
    if (cond > 15) return EINVAL;
    const at = p.cb.offset();
    const word: u32 = if (cond < 0) 0x1400_0000 else 0x5400_0000 | @as(u32, @intCast(cond));
    p.cb.addFixup(.{ .at = at, .base = at, .label = label, .kind = if (cond < 0) .a64_imm26 else .a64_imm19 }) catch |e| return status(e);
    p.cb.ensure(4) catch |e| return status(e);
    p.cb.put32(word);
    return OK;
}

// ---- fast path: opcode ids instead of mnemonic strings (include/pyrlyn_jit_fast.h) ----------
// Resolve a mnemonic once with `pjit_*_opcode`, then emit by id: no string hashing per
// instruction. Ids are stable for one library build (they index the generated tables); the
// string entry points stay for compatibility.

const a64_mn_count = @typeInfo(a64.Mnemonic).@"enum".field_names.len;
// a64.Mnemonic is an auto-numbered exhaustive enum (generated table), so ids are 0..len-1.

/// AArch64 opcode id for `mnemonic`, or -1 if unknown.
export fn pjit_a64_opcode(mnemonic: ?[*:0]const u8) i32 {
    const mn = lookup(a64_map, mnemonic) orelse return -1;
    return @intCast(@backingInt(mn));
}

export fn pjit_a64_emit_op(b: ?*Buffer, opcode: u32, ops: ?[*]const A64Op, n: usize) c_int {
    const p = b orelse return EINVAL;
    // Mnemonic ids are dense (0..len-1): a range check, no table search.
    if (opcode >= a64_mn_count) return @backingInt(Status.unknown_mnemonic);
    const mn: a64.Mnemonic = @fromBackingInt(@as(@typeInfo(a64.Mnemonic).@"enum".tag_type, @intCast(opcode)));
    if (n > 8 or (n > 0 and ops == null)) return EINVAL;
    var conv: [8]a64.Op = undefined;
    for (0..n) |i| conv[i] = a64Op(ops.?[i]) orelse return EINVAL;
    const word = a64.encodeMn(mn, conv[0..n]) orelse return status(error.NoMatchingForm);
    p.cb.ensure(4) catch |e| return status(e);
    p.cb.put32(word);
    return OK;
}

const gp_decls = @typeInfo(gp.tbl.table).@"struct".decl_names;
const gp_forms: [gp_decls.len][]const gp.Form = blk: {
    @setEvalBranchQuota(2_000_000);
    var arr: [gp_decls.len][]const gp.Form = undefined;
    for (gp_decls, 0..) |d, i| arr[i] = &@field(gp.tbl.table, d);
    break :blk arr;
};
const gp_ids = std.StaticStringMap(u32).initComptime(blk: {
    @setEvalBranchQuota(2_000_000);
    var kvs: [gp_decls.len]struct { []const u8, u32 } = undefined;
    for (gp_decls, 0..) |d, i| kvs[i] = .{ d, i };
    break :blk kvs;
});

/// x86-64 GP opcode id for `mnemonic`, or -1 if unknown.
export fn pjit_x64_opcode(mnemonic: ?[*:0]const u8) i32 {
    const id = lookup(gp_ids, mnemonic) orelse return -1;
    return @intCast(id);
}

export fn pjit_x64_emit_op(b: ?*Buffer, opcode: u32, ops: ?[*]const X64Op, n: usize) c_int {
    const p = b orelse return EINVAL;
    if (opcode >= gp_forms.len) return @backingInt(Status.unknown_mnemonic);
    if (n > 4 or (n > 0 and ops == null)) return EINVAL;
    var conv: [4]gp.Op = undefined;
    for (0..n) |i| conv[i] = x64Op(ops.?[i]) orelse return EINVAL;
    gp.emitOps(&p.cb, gp_forms[opcode], conv[0..n]) catch |e| return status(e);
    return OK;
}

// ---- shared executable pool (core/jit_pool.zig) ---------------------------------------------

const Pool = struct { p: pool.Pool };

export fn pjit_pool_new(chunk_size: usize, out: ?*?*Pool) c_int {
    const o = out orelse return EINVAL;
    const h = gpa.create(Pool) catch return status(error.OutOfMemory);
    h.* = .{ .p = pool.Pool.init(gpa, .{ .chunk_size = if (chunk_size == 0) 256 * 1024 else chunk_size }) };
    o.* = h;
    return OK;
}

export fn pjit_pool_free(p: ?*Pool) void {
    const h = p orelse return;
    h.p.deinit();
    gpa.destroy(h);
}

/// Appends finalized code to the pool; `*out` is the executable address (valid until
/// `pjit_pool_free`).
export fn pjit_pool_add(p: ?*Pool, b: ?*const Buffer, out: ?*?*const anyopaque) c_int {
    const h = p orelse return EINVAL;
    const bp = b orelse return EINVAL;
    const o = out orelse return EINVAL;
    if (bp.cb.fixups.items.len != 0) return status(error.UnboundLabel);
    const f = h.p.add(&bp.cb) catch |e| return status(e);
    o.* = @ptrCast(f.addr);
    return OK;
}

// ---- x86-64 (GP instructions) ---------------------------------------------------------------

pub const X64Op = extern struct {
    kind: u8,
    reg: u8,
    size: u8,
    high: u8,
    base: u8,
    index: u8,
    scale: u8,
    rip: u8,
    disp: i32,
    _pad: u32 = 0,
    imm: i64,
};

fn x64Op(o: X64Op) ?gp.Op {
    const size = en(gp.Size, o.size) orelse return null;
    return switch (o.kind) {
        0 => .{ .reg = .{ .id = en(x86.Gp, o.reg) orelse return null, .size = size, .high = o.high != 0 } },
        1 => .{ .mem = .{
            .base = if (o.base == 0xFF) null else en(x86.Gp, o.base) orelse return null,
            .index = if (o.index == 0xFF) null else en(x86.Gp, o.index) orelse return null,
            .scale = en(x86.Scale, o.scale) orelse return null,
            .disp = o.disp,
            .size = size,
            .rip = o.rip != 0,
        } },
        2 => .{ .imm = o.imm },
        else => null,
    };
}

export fn pjit_x64_emit(b: ?*Buffer, mnemonic: ?[*:0]const u8, ops: ?[*]const X64Op, n: usize) c_int {
    const p = b orelse return EINVAL;
    const forms = lookup(gp_map, mnemonic) orelse return @backingInt(Status.unknown_mnemonic);
    if (n > 4 or (n > 0 and ops == null)) return EINVAL;
    var conv: [4]gp.Op = undefined;
    for (0..n) |i| conv[i] = x64Op(ops.?[i]) orelse return EINVAL;
    gp.emitOps(&p.cb, forms, conv[0..n]) catch |e| return status(e);
    return OK;
}

/// `jmp label` (cc < 0) or `j<cc> label` (cc 0..15, x86 condition code), rel32.
export fn pjit_x64_jump(b: ?*Buffer, cc: c_int, label: u32) c_int {
    const p = b orelse return EINVAL;
    if (cc > 15) return EINVAL;
    p.cb.ensure(6) catch |e| return status(e);
    const start = p.cb.offset();
    const len: u32 = if (cc < 0) 5 else 6;
    p.cb.addFixup(.{ .at = start + len - 4, .base = start + len, .label = label, .kind = .rel32 }) catch |e| return status(e);
    if (cc < 0) p.cb.put8(0xE9) else {
        p.cb.put8(0x0F);
        p.cb.put8(0x80 | @as(u8, @intCast(cc)));
    }
    p.cb.put32(0);
    return OK;
}

// ---- executable memory ----------------------------------------------------------------------

const Func = struct { f: rt.Function };

/// Copies finalized code into W^X executable memory.
export fn pjit_jit_add(b: ?*const Buffer, out: ?*?*Func) c_int {
    const p = b orelse return EINVAL;
    const o = out orelse return EINVAL;
    if (p.cb.fixups.items.len != 0) return status(error.UnboundLabel); // not finalized
    const f = rt.add(&p.cb) catch |e| return status(e);
    const h = gpa.create(Func) catch {
        f.release();
        return status(error.OutOfMemory);
    };
    h.* = .{ .f = f };
    o.* = h;
    return OK;
}

export fn pjit_func_ptr(f: ?*const Func) ?*const anyopaque {
    const p = f orelse return null;
    return p.f.mem.ptr;
}

export fn pjit_func_free(f: ?*Func) void {
    const p = f orelse return;
    p.f.release();
    gpa.destroy(p);
}

test "c api: a64 add + ret, labels, errors" {
    var b: ?*Buffer = null;
    try std.testing.expectEqual(OK, pjit_buffer_new(&b));
    defer pjit_buffer_free(b);
    const X = struct {
        fn r(id: u8) A64Op {
            return .{ .kind = 0, .reg = id, .sub = 1, .arr = 0, .et = 0, .idx = -1, .index = 0xFF, .index64 = 0, .ext_op = 0, .ext_n = 0, .has_ext = 0, .imm = 0, .fimm = 0 };
        }
    };
    const ops = [_]A64Op{ X.r(0), X.r(0), X.r(1) };
    try std.testing.expectEqual(OK, pjit_a64_emit(b, "add", &ops, 3));
    try std.testing.expectEqual(@backingInt(Status.unknown_mnemonic), pjit_a64_emit(b, "nope", &ops, 3));
    var bad = X.r(0);
    bad.reg = 200;
    try std.testing.expectEqual(EINVAL, pjit_a64_emit(b, "add", &[_]A64Op{ bad, bad, bad }, 3));
    var l: u32 = 0;
    try std.testing.expectEqual(OK, pjit_label_new(b, &l));
    try std.testing.expectEqual(OK, pjit_a64_branch(b, -1, l));
    try std.testing.expectEqual(@backingInt(Status.unbound_label), pjit_finalize(b));
    try std.testing.expectEqual(OK, pjit_label_bind(b, l));
    try std.testing.expectEqual(OK, pjit_finalize(b));
    var n: usize = 0;
    const c = pjit_buffer_code(b, &n).?;
    try std.testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0x01, 0x8B, 0x01, 0x00, 0x00, 0x14 }, c[0..n]);
    try std.testing.expectEqual(EINVAL, pjit_a64_branch(b, 99, l));
    try std.testing.expectEqual(@backingInt(Status.invalid_label), pjit_label_bind(b, 12345));
}

test "c api: opcode fast path emits the same bytes as the mnemonic path; pool" {
    var b: ?*Buffer = null;
    try std.testing.expectEqual(OK, pjit_buffer_new(&b));
    defer pjit_buffer_free(b);
    const R = struct {
        fn r(id: u8) A64Op {
            return .{ .kind = 0, .reg = id, .sub = 1, .arr = 0, .et = 0, .idx = -1, .index = 0xFF, .index64 = 0, .ext_op = 0, .ext_n = 0, .has_ext = 0, .imm = 0, .fimm = 0 };
        }
    };
    const ops = [_]A64Op{ R.r(0), R.r(0), R.r(1) };
    const add = pjit_a64_opcode("add");
    try std.testing.expect(add >= 0);
    try std.testing.expectEqual(@as(i32, -1), pjit_a64_opcode("nope"));
    try std.testing.expectEqual(OK, pjit_a64_emit(b, "add", &ops, 3));
    try std.testing.expectEqual(OK, pjit_a64_emit_op(b, @intCast(add), &ops, 3));
    try std.testing.expectEqual(@backingInt(Status.unknown_mnemonic), pjit_a64_emit_op(b, 0xFFFF_FFFF, &ops, 3));
    var n: usize = 0;
    const c = pjit_buffer_code(b, &n).?;
    try std.testing.expectEqual(@as(usize, 8), n);
    try std.testing.expectEqualSlices(u8, c[0..4], c[4..8]);

    pjit_buffer_reset(b);
    const X = struct {
        fn r(id: u8) X64Op {
            return .{ .kind = 0, .reg = id, .size = 8, .high = 0, .base = 0xFF, .index = 0xFF, .scale = 0, .rip = 0, .disp = 0, .imm = 0 };
        }
    };
    const xo = [_]X64Op{ X.r(0), X.r(1) };
    const xadd = pjit_x64_opcode("add");
    try std.testing.expect(xadd >= 0);
    try std.testing.expectEqual(OK, pjit_x64_emit(b, "add", &xo, 2));
    try std.testing.expectEqual(OK, pjit_x64_emit_op(b, @intCast(xadd), &xo, 2));
    const c2 = pjit_buffer_code(b, &n).?;
    try std.testing.expect(n % 2 == 0 and n > 0);
    try std.testing.expectEqualSlices(u8, c2[0 .. n / 2], c2[n / 2 .. n]);

    // pool: `ret` on the host
    pjit_buffer_reset(b);
    if (builtin.cpu.arch == .aarch64) {
        try std.testing.expectEqual(OK, pjit_buffer_emit_bytes(b, &[_]u8{ 0xC0, 0x03, 0x5F, 0xD6 }, 4));
    } else try std.testing.expectEqual(OK, pjit_buffer_emit_bytes(b, &[_]u8{0xC3}, 1));
    try std.testing.expectEqual(OK, pjit_finalize(b));
    var pl: ?*Pool = null;
    try std.testing.expectEqual(OK, pjit_pool_new(0, &pl));
    defer pjit_pool_free(pl);
    var f1: ?*const anyopaque = null;
    var f2: ?*const anyopaque = null;
    try std.testing.expectEqual(OK, pjit_pool_add(pl, b, &f1));
    try std.testing.expectEqual(OK, pjit_pool_add(pl, b, &f2));
    try std.testing.expect(@intFromPtr(f2.?) - @intFromPtr(f1.?) == 16);
    if (builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .x86_64) {
        const F = *const fn () callconv(.c) void;
        @as(F, @ptrCast(@alignCast(f2.?)))();
    }
}
