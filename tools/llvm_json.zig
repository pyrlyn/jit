//! Helpers for reading `llvm-tblgen --dump-json` output (LLVM TableGen records as JSON).
const std = @import("std");
pub const Value = std.json.Value;
pub const Obj = std.json.ObjectMap;

pub const Db = struct {
    root: Obj,

    pub fn load(arena: std.mem.Allocator, io: std.Io, path: []const u8) !Db {
        const src = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited);
        const v = try std.json.parseFromSliceLeaky(Value, arena, src, .{});
        return .{ .root = v.object };
    }

    pub fn get(db: Db, name: []const u8) ?Obj {
        const v = db.root.get(name) orelse return null;
        return if (v == .object) v.object else null;
    }

    /// Names of all records deriving from `class`, in record-name order.
    pub fn instancesOf(db: Db, class: []const u8) []const Value {
        const io = db.root.get("!instanceof").?.object;
        return (io.get(class) orelse return &.{}).array.items;
    }
};

pub fn str(o: Obj, key: []const u8) []const u8 {
    const v = o.get(key) orelse return "";
    return switch (v) {
        .string => |s| s,
        else => "",
    };
}

/// Name of a record-valued field (`{"def": "X", "kind": "def"}`).
pub fn def(o: Obj, key: []const u8) []const u8 {
    const v = o.get(key) orelse return "";
    return defName(v);
}

pub fn defName(v: Value) []const u8 {
    return switch (v) {
        .object => |x| if (x.get("def")) |d| (if (d == .string) d.string else "") else "",
        .string => |s| s,
        else => "",
    };
}

pub fn int(o: Obj, key: []const u8) i64 {
    const v = o.get(key) orelse return 0;
    return switch (v) {
        .integer => |i| i,
        .bool => |b| @intFromBool(b),
        else => 0,
    };
}

/// A `bits<N>` field (JSON array, least significant bit first) as an integer; non-constant
/// bits (variable references) read as 0.
pub fn bits(o: Obj, key: []const u8) u64 {
    const v = o.get(key) orelse return 0;
    if (v != .array) return 0;
    var r: u64 = 0;
    for (v.array.items, 0..) |b, i| if (b == .integer and b.integer == 1) {
        r |= @as(u64, 1) << @intCast(i);
    };
    return r;
}

pub fn list(o: Obj, key: []const u8) []const Value {
    const v = o.get(key) orelse return &.{};
    return if (v == .array) v.array.items else &.{};
}

pub fn hasDef(o: Obj, key: []const u8, name: []const u8) bool {
    for (list(o, key)) |p| if (std.mem.eql(u8, defName(p), name)) return true;
    return false;
}

pub const DagArg = struct { class: []const u8, name: []const u8 };

/// Arguments of a dag field such as `InOperandList` (`(ins GR32:$src1, i32imm:$src2)`).
pub fn dagArgs(arena: std.mem.Allocator, o: Obj, key: []const u8) ![]DagArg {
    const v = o.get(key) orelse return &.{};
    if (v != .object) return &.{};
    const args = v.object.get("args") orelse return &.{};
    var out: std.ArrayList(DagArg) = .empty;
    for (args.array.items) |a| {
        const pair = a.array.items;
        const nm: []const u8 = if (pair.len > 1 and pair[1] == .string) pair[1].string else "";
        try out.append(arena, .{ .class = defName(pair[0]), .name = nm });
    }
    return out.items;
}

/// The variant of an LLVM asm string (`{att|intel}` alternatives; a `{x}` with one alternative
/// is AT&T-only). Variant 1 is Intel syntax. Literal braces inside a variant (`{${mask}}`, `{z}`)
/// are kept.
pub fn asmVariant(arena: std.mem.Allocator, s: []const u8, variant: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c == '$' and i + 1 < s.len and s[i + 1] == '{') {
            const e = std.mem.indexOfScalarPos(u8, s, i, '}').?;
            try out.appendSlice(arena, s[i .. e + 1]);
            i = e + 1;
        } else if (c == '\\' and i + 1 < s.len) {
            try out.append(arena, s[i + 1]);
            i += 2;
        } else if (c == '{') {
            // variant block: split on top-level '|'
            var depth: usize = 0;
            var j = i + 1;
            var alt: usize = 0;
            var start = j;
            var picked: ?[]const u8 = null;
            while (j < s.len) : (j += 1) {
                const d = s[j];
                if (d == '$' and j + 1 < s.len and s[j + 1] == '{') {
                    j = std.mem.indexOfScalarPos(u8, s, j, '}').?;
                } else if (d == '{') {
                    depth += 1;
                } else if (d == '}') {
                    if (depth == 0) break;
                    depth -= 1;
                } else if (d == '|' and depth == 0) {
                    if (alt == variant) picked = s[start..j];
                    alt += 1;
                    start = j + 1;
                }
            }
            if (alt == variant) picked = s[start..j];
            if (picked) |p| try out.appendSlice(arena, try asmVariant(arena, p, variant));
            i = j + 1;
        } else {
            try out.append(arena, c);
            i += 1;
        }
    }
    return out.items;
}

/// Tied operand constraints (`$src1 = $dst`): returns the pairs.
pub fn tiedPairs(arena: std.mem.Allocator, o: Obj) ![][2][]const u8 {
    var out: std.ArrayList([2][]const u8) = .empty;
    var it = std.mem.splitScalar(u8, str(o, "Constraints"), ',');
    while (it.next()) |c| {
        const eq = std.mem.indexOfScalar(u8, c, '=') orelse continue;
        const a = std.mem.trim(u8, c[0..eq], " $\t");
        const b = std.mem.trim(u8, c[eq + 1 ..], " $\t");
        if (std.mem.startsWith(u8, b, "@earlyclobber")) continue;
        try out.append(arena, .{ a, b });
    }
    return out.items;
}
