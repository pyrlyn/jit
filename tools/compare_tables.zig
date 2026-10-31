//! Compare two generated encoder tables (`*_table.zig`) form by form.
//!
//!     zig run tools/compare_tables.zig -- OLD.zig NEW.zig
//!     zig run tools/compare_tables.zig -- --exact A B     (byte-identical files, for check-tables)
//!
//! Every `pub const NAME = [_]Form{ ... };` block inside `pub const table = struct` is read as a
//! list of forms (one `.{ ... },` line each; comment lines are ignored). Reports mnemonics only in
//! one file, forms only in one file, and mnemonics whose forms match as a set but not in order
//! (order decides ties between equally long encodings). `.mode = .x64` on REX.W forms is ignored
//! (such forms are rejected outside 64-bit mode anyway). In vec_table.zig forms, `.w = 2` (WIG) is
//! read as `.w = 0` (both encode W0) and `.n` (disp8*N) is ignored on forms without a memory
//! operand (only a displacement uses it). Exit code 1 if anything differs.
const std = @import("std");

const Table = std.StringArrayHashMapUnmanaged(std.ArrayList([]const u8));

fn load(gpa: std.mem.Allocator, src: []const u8) !Table {
    var t: Table = .empty;
    var cur: ?[]const u8 = null;
    var in_table = false;
    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r\t");
        if (std.mem.startsWith(u8, line, "pub const table = struct")) {
            in_table = true;
            continue;
        }
        if (!in_table) continue;
        if (std.mem.startsWith(u8, line, "pub const ")) {
            const rest = line["pub const ".len..];
            const eq = std.mem.indexOf(u8, rest, " = ") orelse continue;
            var name = rest[0..eq];
            if (name.len > 3 and name[0] == '@') name = name[2 .. name.len - 1];
            cur = name;
            const gop = try t.getOrPut(gpa, name);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            continue;
        }
        if (std.mem.startsWith(u8, line, "};")) {
            cur = null;
            continue;
        }
        if (cur) |c| if (std.mem.startsWith(u8, line, ".{")) {
            const f = try normalize(gpa, std.mem.trimEnd(u8, line, ","));
            try t.getPtr(c).?.append(gpa, f);
        };
    }
    return t;
}

/// The `.ops = &.{...}` part of a form line (operand signature).
fn opsOf(f: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, f, ".ops = &.{") orelse return "";
    var depth: usize = 0;
    var i = at + ".ops = &.".len;
    while (i < f.len) : (i += 1) switch (f[i]) {
        '{' => depth += 1,
        '}' => {
            depth -= 1;
            if (depth == 0) return f[at .. i + 1];
        },
        else => {},
    };
    return f[at..];
}

fn listLen(f: []const u8, key: []const u8) usize {
    const at = std.mem.indexOf(u8, f, key) orelse return 0;
    const start = at + key.len;
    const end = std.mem.indexOfScalarPos(u8, f, start, '}') orelse return 0;
    const body = std.mem.trim(u8, f[start..end], " ");
    if (body.len == 0) return 0;
    return std.mem.count(u8, body, ",") + 1;
}

fn intField(f: []const u8, key: []const u8) usize {
    const at = std.mem.indexOf(u8, f, key) orelse return 0;
    var i = at + key.len;
    var v: usize = 0;
    while (i < f.len and std.ascii.isDigit(f[i])) : (i += 1) v = v * 10 + (f[i] - '0');
    return v;
}

/// Static encoded length of an x86 form (prefix + opcode + ModRM + immediate bytes); 0 for formats
/// without these fields (AArch64: every form is 4 bytes).
fn staticLen(f: []const u8) usize {
    return listLen(f, ".pre = &.{") + listLen(f, ".op = &.{") +
        @intFromBool(std.mem.indexOf(u8, f, ".modrm = true") != null) + intField(f, ".imm = ") +
        @intFromBool(std.mem.indexOf(u8, f, ".rexw = true") != null);
}

/// REX.W forms cannot be encoded outside 64-bit mode, so `.mode = .x64` on them is redundant.
fn normalize(gpa: std.mem.Allocator, f0: []const u8) ![]const u8 {
    const f = try normalizeVec(gpa, f0);
    if (std.mem.indexOf(u8, f, ".rexw = true") == null) return f;
    const tag = ", .mode = .x64";
    const at = std.mem.indexOf(u8, f, tag) orelse return f;
    return std.mem.concat(gpa, u8, &.{ f[0..at], f[at + tag.len ..] });
}

/// vec_table.zig: W-ignored encodes as W0; disp8*N only matters with a memory operand.
fn normalizeVec(gpa: std.mem.Allocator, f: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, f, ".enc = .") == null) return f;
    var out = try std.mem.replaceOwned(u8, gpa, f, ".w = 2,", ".w = 0,");
    const has_mem = for ([_][]const u8{ ".k = .mem,", ".k = .vx,", ".k = .vy,", ".k = .vz," }) |k| {
        if (std.mem.indexOf(u8, out, k) != null) break true;
    } else false;
    if (!has_mem) if (std.mem.indexOf(u8, out, ".n = ")) |at| {
        const end = std.mem.indexOfScalarPos(u8, out, at, ',') orelse return out;
        out = try std.mem.concat(gpa, u8, &.{ out[0..at], ".n = 1", out[end..] });
    };
    return out;
}

/// Forms are tried in order and the shortest encoding wins, the earlier form on a tie, so only the
/// relative order of forms with the same operand signature and static length matters.
fn tieOrderEqual(a: []const []const u8, b: []const []const u8) bool {
    for (a) |x| {
        const sig = opsOf(x);
        const len = staticLen(x);
        var i: usize = 0;
        var j: usize = 0;
        while (true) {
            while (i < a.len and !(std.mem.eql(u8, opsOf(a[i]), sig) and staticLen(a[i]) == len)) i += 1;
            while (j < b.len and !(std.mem.eql(u8, opsOf(b[j]), sig) and staticLen(b[j]) == len)) j += 1;
            if (i == a.len or j == b.len) {
                if (i != a.len or j != b.len) return false;
                break;
            }
            if (!std.mem.eql(u8, a[i], b[j])) return false;
            i += 1;
            j += 1;
        }
    }
    return true;
}

fn lessStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len == 4 and std.mem.eql(u8, args[1], "--exact")) {
        const cwd = std.Io.Dir.cwd();
        const x = try cwd.readFileAlloc(io, args[2], gpa, .unlimited);
        const y = try cwd.readFileAlloc(io, args[3], gpa, .unlimited);
        if (!std.mem.eql(u8, x, y)) {
            std.debug.print("{s} differs from {s}: run `zig build gen-tables -Dllvm-src=...`\n", .{ args[2], args[3] });
            std.process.exit(1);
        }
        return;
    }
    if (args.len != 3) {
        std.debug.print("usage: compare_tables OLD.zig NEW.zig\n", .{});
        std.process.exit(2);
    }
    const cwd = std.Io.Dir.cwd();
    const a = try load(gpa, try cwd.readFileAlloc(io, args[1], gpa, .unlimited));
    const b = try load(gpa, try cwd.readFileAlloc(io, args[2], gpa, .unlimited));
    var only_a: usize = 0;
    var only_b: usize = 0;
    var diff_forms: usize = 0;
    var order_only: usize = 0;
    var reordered: usize = 0;
    var same: usize = 0;
    var forms_a: usize = 0;
    var forms_b: usize = 0;
    for (a.keys()) |k| forms_a += a.get(k).?.items.len;
    for (b.keys()) |k| forms_b += b.get(k).?.items.len;
    for (a.keys()) |k| if (!b.contains(k)) {
        only_a += 1;
        std.debug.print("- mnemonic {s} ({d} forms) only in OLD\n", .{ k, a.get(k).?.items.len });
    };
    for (b.keys()) |k| if (!a.contains(k)) {
        only_b += 1;
        std.debug.print("+ mnemonic {s} ({d} forms) only in NEW\n", .{ k, b.get(k).?.items.len });
    };
    for (a.keys()) |k| {
        const fb = b.get(k) orelse continue;
        const fa = a.get(k).?;
        if (fa.items.len == fb.items.len) {
            var eq = true;
            for (fa.items, fb.items) |x, y| if (!std.mem.eql(u8, x, y)) {
                eq = false;
                break;
            };
            if (eq) {
                same += 1;
                continue;
            }
        }
        const sa = try gpa.dupe([]const u8, fa.items);
        const sb = try gpa.dupe([]const u8, fb.items);
        std.mem.sort([]const u8, sa, {}, lessStr);
        std.mem.sort([]const u8, sb, {}, lessStr);
        var i: usize = 0;
        var j: usize = 0;
        var any = false;
        while (i < sa.len or j < sb.len) {
            const o: std.math.Order = if (i == sa.len) .gt else if (j == sb.len) .lt else std.mem.order(u8, sa[i], sb[j]);
            switch (o) {
                .eq => {
                    i += 1;
                    j += 1;
                },
                .lt => {
                    std.debug.print("{s}: - {s}\n", .{ k, sa[i] });
                    i += 1;
                    any = true;
                },
                .gt => {
                    std.debug.print("{s}: + {s}\n", .{ k, sb[j] });
                    j += 1;
                    any = true;
                },
            }
        }
        if (any) diff_forms += 1 else if (tieOrderEqual(fa.items, fb.items)) {
            reordered += 1;
        } else {
            order_only += 1;
            std.debug.print("{s}: same forms, different order among forms with equal operands\n", .{k});
        }
    }
    std.debug.print(
        "OLD: {d} mnemonics / {d} forms, NEW: {d} / {d}; identical {d}, reordered (ties kept) {d}, tie order differs {d}, differing {d}, only-OLD {d}, only-NEW {d}\n",
        .{ a.count(), forms_a, b.count(), forms_b, same, reordered, order_only, diff_forms, only_a, only_b },
    );
    if (only_a + only_b + diff_forms + order_only != 0) std.process.exit(1);
}
