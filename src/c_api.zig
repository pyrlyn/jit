//! Stable C ABI (see include/zigjit.h). Every entry point validates its arguments and returns a
//! `zj_status`; no Zig panics or UB are reachable through bad input from C.
const std = @import("std");
const builtin = @import("builtin");
const cbm = @import("core/code_buffer.zig");
const rt = @import("core/jit_runtime.zig");
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
    return @intFromEnum(s);
}
const OK: c_int = 0;
const EINVAL: c_int = @intFromEnum(Status.invalid_argument);

const gpa = std.heap.c_allocator;

const Buffer = struct { cb: cbm.CodeBuffer };

export fn zj_version() u32 {
    return version;
}

export fn zj_status_str(s: c_int) [*:0]const u8 {
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

export fn zj_buffer_new(out: ?*?*Buffer) c_int {
    const o = out orelse return EINVAL;
    const b = gpa.create(Buffer) catch return status(error.OutOfMemory);
    b.* = .{ .cb = cbm.CodeBuffer.init(gpa) };
    o.* = b;
    return OK;
}

export fn zj_buffer_free(b: ?*Buffer) void {
    const p = b orelse return;
    p.cb.deinit();
    gpa.destroy(p);
}

export fn zj_buffer_reset(b: ?*Buffer) void {
    if (b) |p| p.cb.reset();
}

export fn zj_buffer_code(b: ?*const Buffer, len: ?*usize) ?[*]const u8 {
    const p = b orelse return null;
    if (len) |l| l.* = p.cb.code().len;
    return p.cb.code().ptr;
}

export fn zj_buffer_emit_bytes(b: ?*Buffer, bytes: ?[*]const u8, n: usize) c_int {
    const p = b orelse return EINVAL;
    if (n == 0) return OK;
    const src = bytes orelse return EINVAL;
    p.cb.emitBytes(src[0..n]) catch |e| return status(e);
    return OK;
}

export fn zj_label_new(b: ?*Buffer, out: ?*u32) c_int {
    const p = b orelse return EINVAL;
    const o = out orelse return EINVAL;
    const l = p.cb.newLabel() catch |e| return status(e);
    o.* = l.id;
    return OK;
}

export fn zj_label_bind(b: ?*Buffer, label: u32) c_int {
    const p = b orelse return EINVAL;
    p.cb.bind(.{ .id = label }) catch |e| return status(e);
    return OK;
}

export fn zj_finalize(b: ?*Buffer) c_int {
    const p = b orelse return EINVAL;
    p.cb.finalize() catch |e| return status(e);
    return OK;
}

// ---- mnemonic lookup ------------------------------------------------------------------------

fn FormMap(comptime T: type, comptime F: type) type {
    @setEvalBranchQuota(2_000_000);
    const decls = @typeInfo(T).@"struct".decls;
    var kvs: [decls.len]struct { []const u8, []const F } = undefined;
    for (decls, 0..) |d, i| kvs[i] = .{ d.name, &@field(T, d.name) };
    const final = kvs;
    return struct {
        const map = std.StaticStringMap([]const F).initComptime(final);
    };
}
const a64_map = FormMap(a64.tbl.table, a64.Form).map;
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
    _pad: [5]u8 = .{0} ** 5,
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

export fn zj_a64_emit(b: ?*Buffer, mnemonic: ?[*:0]const u8, ops: ?[*]const A64Op, n: usize) c_int {
    const p = b orelse return EINVAL;
    const forms = lookup(a64_map, mnemonic) orelse return @intFromEnum(Status.unknown_mnemonic);
    if (n > 8 or (n > 0 and ops == null)) return EINVAL;
    var conv: [8]a64.Op = undefined;
    for (0..n) |i| conv[i] = a64Op(ops.?[i]) orelse return EINVAL;
    const word = a64.encodeAny(forms, conv[0..n]) orelse return status(error.NoMatchingForm);
    p.cb.ensure(4) catch |e| return status(e);
    p.cb.put32(word);
    return OK;
}

/// `b label` (cond < 0) or `b.<cond> label` (cond 0..15).
export fn zj_a64_branch(b: ?*Buffer, cond: c_int, label: u32) c_int {
    const p = b orelse return EINVAL;
    if (cond > 15) return EINVAL;
    const at = p.cb.offset();
    const word: u32 = if (cond < 0) 0x1400_0000 else 0x5400_0000 | @as(u32, @intCast(cond));
    p.cb.addFixup(.{ .at = at, .base = at, .label = label, .kind = if (cond < 0) .a64_imm26 else .a64_imm19 }) catch |e| return status(e);
    p.cb.ensure(4) catch |e| return status(e);
    p.cb.put32(word);
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

export fn zj_x64_emit(b: ?*Buffer, mnemonic: ?[*:0]const u8, ops: ?[*]const X64Op, n: usize) c_int {
    const p = b orelse return EINVAL;
    const forms = lookup(gp_map, mnemonic) orelse return @intFromEnum(Status.unknown_mnemonic);
    if (n > 4 or (n > 0 and ops == null)) return EINVAL;
    var conv: [4]gp.Op = undefined;
    for (0..n) |i| conv[i] = x64Op(ops.?[i]) orelse return EINVAL;
    gp.emitOps(&p.cb, forms, conv[0..n]) catch |e| return status(e);
    return OK;
}

/// `jmp label` (cc < 0) or `j<cc> label` (cc 0..15, x86 condition code), rel32.
export fn zj_x64_jump(b: ?*Buffer, cc: c_int, label: u32) c_int {
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
export fn zj_jit_add(b: ?*const Buffer, out: ?*?*Func) c_int {
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

export fn zj_func_ptr(f: ?*const Func) ?*const anyopaque {
    const p = f orelse return null;
    return p.f.mem.ptr;
}

export fn zj_func_free(f: ?*Func) void {
    const p = f orelse return;
    p.f.release();
    gpa.destroy(p);
}

test "c api: a64 add + ret, labels, errors" {
    var b: ?*Buffer = null;
    try std.testing.expectEqual(OK, zj_buffer_new(&b));
    defer zj_buffer_free(b);
    const X = struct {
        fn r(id: u8) A64Op {
            return .{ .kind = 0, .reg = id, .sub = 1, .arr = 0, .et = 0, .idx = -1, .index = 0xFF, .index64 = 0, .ext_op = 0, .ext_n = 0, .has_ext = 0, .imm = 0, .fimm = 0 };
        }
    };
    const ops = [_]A64Op{ X.r(0), X.r(0), X.r(1) };
    try std.testing.expectEqual(OK, zj_a64_emit(b, "add", &ops, 3));
    try std.testing.expectEqual(@intFromEnum(Status.unknown_mnemonic), zj_a64_emit(b, "nope", &ops, 3));
    var bad = X.r(0);
    bad.reg = 200;
    try std.testing.expectEqual(EINVAL, zj_a64_emit(b, "add", &[_]A64Op{ bad, bad, bad }, 3));
    var l: u32 = 0;
    try std.testing.expectEqual(OK, zj_label_new(b, &l));
    try std.testing.expectEqual(OK, zj_a64_branch(b, -1, l));
    try std.testing.expectEqual(@intFromEnum(Status.unbound_label), zj_finalize(b));
    try std.testing.expectEqual(OK, zj_label_bind(b, l));
    try std.testing.expectEqual(OK, zj_finalize(b));
    var n: usize = 0;
    const c = zj_buffer_code(b, &n).?;
    try std.testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0x01, 0x8B, 0x01, 0x00, 0x00, 0x14 }, c[0..n]);
    try std.testing.expectEqual(EINVAL, zj_a64_branch(b, 99, l));
    try std.testing.expectEqual(@intFromEnum(Status.invalid_label), zj_label_bind(b, 12345));
}
