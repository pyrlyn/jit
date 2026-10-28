//! Robustness fuzzing: arbitrary operands / buffer operations must give an error or valid
//! output, never a panic or out-of-bounds access (Debug/ReleaseSafe safety checks are on).
//!
//! `zig build test` runs each harness over a fixed PRNG stream (deterministic, CI-friendly).
//! `zig build test --fuzz` drives the same harnesses with Zig's coverage-guided fuzzer.
const std = @import("std");
const a64 = @import("aarch64/a64.zig");
const gp = @import("x86_64/gp.zig");
const cbm = @import("core/code_buffer.zig");

/// Input source: `std.testing.Smith` (fuzzer) or `Prng` (deterministic runs).
const Prng = struct {
    r: std.Random.DefaultPrng,
    pub fn value(self: *Prng, comptime T: type) T {
        return switch (@typeInfo(T)) {
            .bool => self.r.random().boolean(),
            .int => self.r.random().int(T),
            .float => @bitCast(self.r.random().int(std.meta.Int(.unsigned, @bitSizeOf(T)))),
            else => @compileError("unsupported"),
        };
    }
};

fn pickEnum(src: anytype, comptime E: type) E {
    const vals = comptime std.enums.values(E);
    return vals[src.value(u16) % vals.len];
}

fn smallImm(src: anytype) i64 {
    return switch (src.value(u8) % 4) {
        0 => src.value(i8),
        1 => src.value(i16),
        2 => @as(i64, src.value(u12)) * 16,
        else => src.value(i64),
    };
}

// ---- AArch64 encoder ------------------------------------------------------------------------

const a64_forms = blk: {
    @setEvalBranchQuota(1_000_000);
    const decls = @typeInfo(a64.tbl.table).@"struct".decls;
    var out: [decls.len][]const a64.Form = undefined;
    for (decls, 0..) |d, i| out[i] = &@field(a64.tbl.table, d.name);
    break :blk out;
};

fn a64Gp(src: anytype) a64.Gp {
    return .{ .id = src.value(u6), .x = src.value(bool) };
}

fn a64Op(src: anytype) a64.Op {
    return switch (src.value(u8) % 8) {
        0 => .{ .r = a64Gp(src) },
        1 => .{ .v = .{ .id = src.value(u5), .kind = pickEnum(src, a64.VKind), .arr = pickEnum(src, a64.Arr), .et = pickEnum(src, a64.Et), .idx = if (src.value(bool)) src.value(u4) else null } },
        2 => .{ .imm = smallImm(src) },
        3 => .{ .fimm = src.value(f64) },
        4 => .{ .shift = .{ .op = pickEnum(src, a64.ShiftOp), .n = src.value(u8) } },
        5 => .{ .mem = .{ .base = a64Gp(src), .index = if (src.value(bool)) a64Gp(src) else null, .ext = if (src.value(bool)) .{ .op = pickEnum(src, a64.ShiftOp), .n = src.value(u8) } else null, .off = smallImm(src), .mode = pickEnum(src, a64.MemMode) } },
        6 => .{ .cond = src.value(u4) },
        else => .{ .rel = smallImm(src) },
    };
}

fn a64One(src: anytype) !void {
    const forms = a64_forms[src.value(u16) % a64_forms.len];
    var ops: [6]a64.Op = undefined;
    const n = src.value(u8) % 7;
    for (ops[0..n]) |*o| o.* = a64Op(src);
    _ = a64.encodeAny(forms, ops[0..n]);
}

const GpId = @FieldType(gp.Reg, "id");

// ---- x86-64 GP encoder ----------------------------------------------------------------------

const gp_forms = blk: {
    @setEvalBranchQuota(1_000_000);
    const decls = @typeInfo(gp.tbl.table).@"struct".decls;
    var out: [decls.len][]const gp.Form = undefined;
    for (decls, 0..) |d, i| out[i] = &@field(gp.tbl.table, d.name);
    break :blk out;
};

fn gpOp(src: anytype) gp.Op {
    return switch (src.value(u8) % 3) {
        0 => .{ .reg = .{ .id = pickEnum(src, GpId), .size = pickEnum(src, gp.Size), .high = src.value(u8) % 8 == 0 } },
        1 => .{ .mem = .{ .base = if (src.value(bool)) pickEnum(src, GpId) else null, .index = if (src.value(bool)) pickEnum(src, GpId) else null, .scale = pickEnum(src, @FieldType(gp.Mem, "scale")), .disp = src.value(i32), .size = pickEnum(src, gp.Size), .rip = src.value(u8) % 8 == 0 } },
        else => .{ .imm = smallImm(src) },
    };
}

fn gpOne(src: anytype) !void {
    var buf = cbm.CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    const forms = gp_forms[src.value(u16) % gp_forms.len];
    var ops: [4]gp.Op = undefined;
    const n = src.value(u8) % 5;
    for (ops[0..n]) |*o| o.* = gpOp(src);
    gp.emitOps(&buf, forms, ops[0..n]) catch return;
    if (buf.code().len == 0 or buf.code().len > 15) return error.BadInstructionLength;
}

// ---- CodeBuffer labels / fixups -------------------------------------------------------------

fn bufferOne(src: anytype) !void {
    var buf = cbm.CodeBuffer.init(std.testing.allocator);
    defer buf.deinit();
    buf.max_size = 1 << 16;
    for (0..src.value(u8) % 64) |_| switch (src.value(u8) % 6) {
        0 => _ = buf.newLabel() catch {},
        1 => buf.bind(.{ .id = src.value(u8) % 16 }) catch {},
        2 => buf.addFixup(.{ .at = src.value(u16), .base = src.value(u16), .label = src.value(u8) % 16, .kind = pickEnum(src, cbm.FixupKind) }) catch {},
        3 => {
            var b: [32]u8 = undefined;
            for (&b) |*x| x.* = src.value(u8);
            buf.emitBytes(b[0 .. src.value(u8) % 32]) catch {};
        },
        4 => _ = buf.labelOffset(.{ .id = src.value(u32) }),
        else => {
            const before = try std.testing.allocator.dupe(u8, buf.code());
            defer std.testing.allocator.free(before);
            buf.finalize() catch {
                // A failed finalize must leave the code untouched.
                try std.testing.expectEqualSlices(u8, before, buf.code());
            };
        },
    };
}

// ---- drivers --------------------------------------------------------------------------------

fn runPrng(comptime one: anytype, iters: usize, seed: u64) !void {
    var src: Prng = .{ .r = .init(seed) };
    for (0..iters) |_| try one(&src);
}

fn fuzzEntry(comptime one: anytype) fn (void, *std.testing.Smith) anyerror!void {
    return struct {
        fn f(_: void, s: *std.testing.Smith) anyerror!void {
            try one(s);
        }
    }.f;
}

test "fuzz: a64 encoder never panics" {
    try runPrng(a64One, 200_000, 0xA64);
    try std.testing.fuzz({}, fuzzEntry(a64One), .{});
}

test "fuzz: x86-64 gp encoder never panics; instructions are 1..15 bytes" {
    try runPrng(gpOne, 100_000, 0x86);
    try std.testing.fuzz({}, fuzzEntry(gpOne), .{});
}

test "fuzz: code buffer labels/fixups are bounds-checked" {
    try runPrng(bufferOne, 50_000, 0xB0F);
    try std.testing.fuzz({}, fuzzEntry(bufferOne), .{});
}
