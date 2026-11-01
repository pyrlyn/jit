//! AArch64 forms (src/aarch64/a64_table.zig) from LLVM's AArch64 target description
//! (`llvm-tblgen --dump-json AArch64.td`).
//!
//! Every LLVM `Instruction` record with a 32-bit encoding and every `InstAlias` becomes a form in
//! the shape src/aarch64/a64.zig encodes: the fixed bits, the operand fields (`.n`, `.at`, `.w`,
//! `.src`: bits `src..src+w` of the operand value go to `at..at+w`), the operand list and, for
//! immediates that are not stored as-is, the a64.zig conversion (`ImmFn`). Field positions come
//! from the record's `Inst` bit layout; the operand kinds come from the operand classes (see
//! `single` below for the class -> kind list). Records that differ only in the element-size field
//! (bits 22-23) are merged into one form with element lists (`.t`/`.ta`/`.tb`, `.zt`/`.zta`/`.ztb`).
//!
//! Project decisions that are not LLVM data are the commented lists near the end of this file:
//! aliases LLVM's assembler implements in C++ (`mov #imm`, `lsl #n`, the bitfield aliases, the
//! `sys` aliases), AsmJit compatibility forms, form order hints, and the mnemonics/field names kept
//! so existing code compiles.
const std = @import("std");
const lj = @import("llvm_json.zig");
const Value = lj.Value;
const Obj = lj.Obj;
const Allocator = std.mem.Allocator;
const eql = std.mem.eql;
const startsWith = std.mem.startsWith;
const endsWith = std.mem.endsWith;

pub const Op = struct {
    k: []const u8,
    f: ?[]const u8 = null,
    f2: ?[]const u8 = null,
    arr: ?[]const u8 = null,
    et: ?[]const u8 = null,
    sh: ?[]const u8 = null,
    mode: ?[]const u8 = null,
    mk: ?[]const u8 = null,
    n: ?i64 = null,
    w: ?bool = null,
    fixed: ?i64 = null,
    scale: ?i64 = null,
    signed: ?bool = null,
    opt: ?bool = null,
};

pub const Field = struct { n: []const u8, at: u32, w: u32, src: u32 };

const Lists = struct {
    t: []const []const u8 = &.{},
    ta: []const []const u8 = &.{},
    tb: []const []const u8 = &.{},
    zt: []const []const u8 = &.{},
    zta: []const []const u8 = &.{},
    ztb: []const []const u8 = &.{},
};

pub const Form = struct {
    mn: []const u8,
    bits: u32,
    fields: []const Field,
    ops: []const Op,
    imm: []const u8 = "none",
    args: []const i64 = &.{},
    rec: []const u8,
    lists: Lists = .{},
    inv_cond: bool = false,
};

const Skip = error{Skip};
const Error = Skip || Allocator.Error;

// ---- operand classes -------------------------------------------------------------------------

const KV = struct { []const u8, []const u8 };
fn lookup(tab: []const KV, key: []const u8) ?[]const u8 {
    for (tab) |e| if (eql(u8, e[0], key)) return e[1];
    return null;
}
fn inList(l: []const []const u8, s: []const u8) bool {
    for (l) |x| if (eql(u8, x, s)) return true;
    return false;
}

/// General-purpose register classes -> operand kind (`w`/`x`, `wsp`/`xsp` also accept SP).
const gpr_classes = [_]KV{
    .{ "GPR32", "w" },     .{ "GPR64", "x" },         .{ "GPR32z", "w" },        .{ "GPR64z", "x" },
    .{ "GPR32sp", "wsp" }, .{ "GPR64sp", "xsp" },     .{ "GPR64common", "x" },   .{ "GPR64as32", "w" },
    .{ "GPR32as64", "x" }, .{ "GPR64noip", "x" },     .{ "tcGPR64", "x" },       .{ "GPR32arg", "w" },
    .{ "GPR64arg", "x" },  .{ "GPR64sponly", "xsp" }, .{ "GPR32sponly", "wsp" }, .{ "GPR64sp0", "xsp" },
};
/// Scalar FP/SIMD register classes (`FPR8`..`FPR128` and their `Op`/`asZPR`/`_lo` variants).
fn fprKind(c: []const u8) ?[]const u8 {
    const base = [_]KV{ .{ "FPR8", "sb" }, .{ "FPR16", "sh" }, .{ "FPR32", "ss" }, .{ "FPR64", "sd" }, .{ "FPR128", "sq" } };
    for (base) |b| if (startsWith(u8, c, b[0])) {
        const suf = c[b[0].len..];
        if (suf.len == 0 or eql(u8, suf, "Op") or eql(u8, suf, "asZPR") or eql(u8, suf, "_lo")) return b[1];
    };
    return null;
}
/// Vector register classes (the arrangement comes from the asm string, `$Rd.4s`).
const vreg_classes = [_][]const u8{ "V64", "V128", "V128_lo", "V128_0to7", "V64_lo", "V64_0to7", "FPR128_lo", "FPR64_lo", "FPR32_lo", "FPR16_lo" };
/// SVE vector register classes -> element type.
const zpr_classes = [_]KV{
    .{ "ZPR8", "b" },    .{ "ZPR16", "h" },   .{ "ZPR32", "s" },   .{ "ZPR64", "d" },   .{ "ZPR128", "q" },
    .{ "ZPR3b8", "b" },  .{ "ZPR3b16", "h" }, .{ "ZPR3b32", "s" }, .{ "ZPR3b64", "d" }, .{ "ZPR4b8", "b" },
    .{ "ZPR4b16", "h" }, .{ "ZPR4b32", "s" }, .{ "ZPR4b64", "d" }, .{ "Z_b", "b" },     .{ "Z_h", "h" },
    .{ "Z_s", "s" },     .{ "Z_d", "d" },     .{ "Z_q", "q" },
};
/// SVE predicate classes with an element type (`Pd.B`).
const ppr_classes = [_]KV{
    .{ "PPR8", "b" },      .{ "PPR16", "h" },      .{ "PPR32", "s" },      .{ "PPR64", "d" },
    .{ "PPRorPNR8", "b" }, .{ "PPRorPNR16", "h" }, .{ "PPRorPNR32", "s" }, .{ "PPRorPNR64", "d" },
};
/// SVE governing predicate classes (`Pg`, `Pg/M`, `Pg/Z`).
const pg_classes = [_][]const u8{ "PPR3bAny", "PPRAny", "PPRorPNRAny", "PPR_3b", "PPR" };

const arrs = [_]KV{
    .{ "8b", "a8b" }, .{ "16b", "a16b" }, .{ "4h", "a4h" }, .{ "8h", "a8h" }, .{ "2s", "a2s" },
    .{ "4s", "a4s" }, .{ "1d", "a1d" },   .{ "2d", "a2d" }, .{ "1q", "a1q" }, .{ "4b", "a4b" },
    .{ "2h", "a2h" }, .{ "2b", "a2b" },   .{ "1s", "a1s" },
};
const list_counts = [_]struct { []const u8, i64 }{ .{ "One", 1 }, .{ "Two", 2 }, .{ "Three", 3 }, .{ "Four", 4 } };

fn log2(n: i64) i64 {
    var v = n;
    var r: i64 = -1;
    while (v > 0) : (v >>= 1) r += 1;
    return r;
}

// ---- tiny pattern helpers --------------------------------------------------------------------

const Cur = struct {
    s: []const u8,
    i: usize = 0,
    fn lit(c: *Cur, t: []const u8) bool {
        if (startsWith(u8, c.s[c.i..], t)) {
            c.i += t.len;
            return true;
        }
        return false;
    }
    fn digits(c: *Cur) ?[]const u8 {
        const st = c.i;
        while (c.i < c.s.len and std.ascii.isDigit(c.s[c.i])) c.i += 1;
        return if (c.i > st) c.s[st..c.i] else null;
    }
    fn word(c: *Cur) ?[]const u8 {
        const st = c.i;
        while (c.i < c.s.len and (std.ascii.isAlphanumeric(c.s[c.i]) or c.s[c.i] == '_')) c.i += 1;
        return if (c.i > st) c.s[st..c.i] else null;
    }
    fn oneOf(c: *Cur, set: []const u8) ?u8 {
        if (c.i < c.s.len and std.mem.indexOfScalar(u8, set, c.s[c.i]) != null) {
            c.i += 1;
            return c.s[c.i - 1];
        }
        return null;
    }
    fn end(c: *Cur) bool {
        return c.i == c.s.len;
    }
};

fn num(s: []const u8) i64 {
    return std.fmt.parseInt(i64, s, 10) catch 0;
}
fn isNum(s: []const u8) bool {
    var c: Cur = .{ .s = s };
    _ = c.lit("-");
    return c.digits() != null and c.end();
}
/// `$name` (whole string) -> name
fn dollarWord(s: []const u8) ?[]const u8 {
    var c: Cur = .{ .s = s };
    if (!c.lit("$")) return null;
    const w = c.word() orelse return null;
    return if (c.end()) w else null;
}

/// Plain immediate classes: the value goes into the field unchanged.
fn plainImm(c: []const u8) ?[]const u8 {
    // t?imm(32|64)?_?0_N, imm0_N(_64b)?, uimmN, i32imm, i64imm, uimmN_32b|_64b, tbz_imm0_31_diag
    {
        var p: Cur = .{ .s = c };
        _ = p.lit("t");
        if (p.lit("imm")) {
            _ = p.lit("32") or p.lit("64");
            _ = p.lit("_");
            if (p.lit("0_") and p.digits() != null and (p.end() or (p.lit("_64b") and p.end()))) return "imm";
        }
    }
    {
        var p: Cur = .{ .s = c };
        if (p.lit("uimm") and p.digits() != null and (p.end() or ((p.lit("_32b") or p.lit("_64b")) and p.end()))) return "imm";
    }
    if (eql(u8, c, "i32imm") or eql(u8, c, "i64imm") or eql(u8, c, "tbz_imm0_31_diag")) return "imm";
    {
        var p: Cur = .{ .s = c };
        if (p.lit("simm") and p.digits() != null and p.lit("_")) {
            if ((p.lit("8b") or p.lit("16b") or p.lit("32b") or p.lit("64b")) and p.end()) return "immS";
        }
    }
    return null;
}

// ---- records ---------------------------------------------------------------------------------

const Bit = union(enum) { none, fixed: u1, v: struct { name: []const u8, idx: u32 } };

/// What `build` needs from an instruction record (or from an alias, rewritten onto its target).
const Rec = struct {
    asm_str: []const u8,
    inst: [32]Bit,
    constraints: []const u8,
    /// operand name -> class, outs then ins
    om: std.StringArrayHashMapUnmanaged([]const u8) = .empty,
};

fn classOf(v: Value) []const u8 {
    if (v != .object) return "?";
    if (v.object.get("def")) |d| if (d == .string) return d.string;
    if (v.object.get("operator")) |o| return lj.defName(o);
    return "?";
}

fn instBits(arr: []const Value) ?[32]Bit {
    if (arr.len != 32) return null;
    var out: [32]Bit = undefined;
    for (arr, 0..) |b, i| {
        out[i] = switch (b) {
            .integer => |x| .{ .fixed = @intCast(x & 1) },
            .object => |o| blk: {
                const kind = lj.str(o, "kind");
                if (eql(u8, kind, "varbit")) break :blk .{ .v = .{ .name = lj.str(o, "var"), .idx = @intCast(lj.int(o, "index")) } };
                if (eql(u8, kind, "var")) break :blk .{ .v = .{ .name = lj.str(o, "var"), .idx = 0 } };
                break :blk .none;
            },
            else => .none,
        };
    }
    return out;
}

fn recordOf(a: Allocator, r: Obj) !?Rec {
    const inst = r.get("Inst") orelse return null;
    if (inst != .array) return null;
    var rec: Rec = .{ .asm_str = lj.str(r, "AsmString"), .inst = instBits(inst.array.items) orelse return null, .constraints = lj.str(r, "Constraints") };
    for ([_][]const u8{ "OutOperandList", "InOperandList" }) |key| {
        for (dagArgs(r, key)) |pv| {
            const p = pairOf(pv);
            try rec.om.put(a, p.name, classOf(p.cls));
        }
    }
    return rec;
}

const Pair = struct { cls: Value, name: []const u8 };

/// `[class, name]` pairs of a dag field (`(ins GPR64sp:$Rn, simm9:$imm)`), with the class value.
fn dagArgs(o: Obj, key: []const u8) []const Value {
    const v = o.get(key) orelse return &.{};
    if (v != .object) return &.{};
    const args = v.object.get("args") orelse return &.{};
    return if (args == .array) args.array.items else &.{};
}
fn pairOf(p: Value) Pair {
    if (p != .array or p.array.items.len == 0) return .{ .cls = .null, .name = "" };
    const it = p.array.items;
    return .{ .cls = it[0], .name = if (it.len > 1 and it[1] == .string) it[1].string else "" };
}

// ---- building one form -----------------------------------------------------------------------

const Split = struct { lo: u32, hi: u32, name: []const u8, base: u32 };

/// Per-record state while its asm operands are read: how the record's `Inst` variables become
/// form fields.
const B = struct {
    a: Allocator,
    rec: *const Rec,
    name: []const u8,
    /// variable -> [(bit position, variable bit)]
    vs: std.StringArrayHashMapUnmanaged(std.ArrayList([2]u32)) = .empty,
    fixed_val: u32 = 0,
    tie: std.StringHashMapUnmanaged([]const u8) = .empty,
    /// variable -> field name
    rename: std.StringHashMapUnmanaged([]const u8) = .empty,
    /// variable -> bit ranges with their own field names (`#0`: dropped, the bit stays 0)
    split: std.StringHashMapUnmanaged([]const Split) = .empty,
    /// fields placed explicitly (their bits are not taken from the variables)
    over: std.ArrayList(Field) = .empty,
    forced: std.AutoArrayHashMapUnmanaged(u32, u1) = .empty,
    imm: []const u8 = "none",
    args: []const i64 = &.{},
    inv_cond: bool = false,
    /// post-index register form that also gets a fixed `#imm` twin (Rm = 31)
    alt_n: ?i64 = null,
    consume_next: bool = false,

    fn cls(b: *B, v: []const u8) ?[]const u8 {
        return b.rec.om.get(v);
    }
    fn ren(b: *B, v: []const u8) []const u8 {
        return b.rename.get(v) orelse v;
    }
    fn setRename(b: *B, v: []const u8, n: []const u8) !void {
        try b.rename.put(b.a, v, n);
    }
    fn setSplit(b: *B, v: []const u8, parts: []const Split) !void {
        try b.split.put(b.a, v, try b.a.dupe(Split, parts));
    }
    fn addOver(b: *B, n: []const u8, at: u32, w: u32, src: u32) !void {
        try b.over.append(b.a, .{ .n = n, .at = at, .w = w, .src = src });
    }
    fn setImm(b: *B, f: []const u8, args: []const i64) !void {
        b.imm = f;
        b.args = try b.a.dupe(i64, args);
    }
    /// Field name for a register variable (a tied operand resolves to the encoded one).
    fn fname(b: *B, v: []const u8) ?[]const u8 {
        if (b.vs.contains(v)) return b.ren(v);
        if (b.tie.get(v)) |t| if (b.vs.contains(t)) return b.ren(t);
        return null;
    }
    fn anyClass(b: *B, comptime pred: fn ([]const u8) bool) bool {
        for (b.rec.om.values()) |c| if (pred(c)) return true;
        return false;
    }
    fn sve(b: *B) bool {
        return b.anyClass(struct {
            fn f(c: []const u8) bool {
                return startsWith(u8, c, "ZPR");
            }
        }.f);
    }
    fn asmMnemonic(b: *B) ![]const u8 {
        const v = try lj.asmVariant(b.a, b.rec.asm_str, 0);
        const t = std.mem.indexOfScalar(u8, v, '\t') orelse v.len;
        const sp = std.mem.indexOfScalar(u8, v[0..t], ' ') orelse t;
        return v[0..sp];
    }
};

fn one(a: Allocator, o: Op) ![]Op {
    const r = try a.alloc(Op, 1);
    r[0] = o;
    return r;
}
fn two(a: Allocator, x: Op, y: Op) ![]Op {
    const r = try a.alloc(Op, 2);
    r[0] = x;
    r[1] = y;
    return r;
}

fn memBaseOk(c: ?[]const u8) bool {
    const s = c orelse return false;
    return eql(u8, s, "GPR64sp") or eql(u8, s, "GPR64sp0") or eql(u8, s, "GPR64common");
}

/// One asm operand token (`$Rd`, `$Rd.4s`, `[$Rn, $imm]!`, `#0`, ...) -> operand specs.
fn token(b: *B, t0: []const u8, nxt: ?[]const u8) Error![]Op {
    const a = b.a;
    if (eql(u8, t0, "#0") or eql(u8, t0, "#0.0")) return one(a, .{ .k = "immfix", .fixed = 0 });
    if (t0.len > 1 and t0[0] == '#' and isNum(t0[1..]) and t0[1] != '-') return one(a, .{ .k = "immfix", .fixed = num(t0[1..]) });
    const t = if (startsWith(u8, t0, "mul $")) t0[4..] else t0;
    // literal register (`chkfeat x16`)
    if (t.len > 1 and (t[0] == 'w' or t[0] == 'x')) {
        var c: Cur = .{ .s = t[1..] };
        if (c.digits() != null and c.end()) return one(a, .{ .k = t[0..1], .fixed = num(t[1..]) });
    }
    if (t.len > 1 and t[0] == '$') {
        var c: Cur = .{ .s = t, .i = 1 };
        if (c.word()) |v| return single(b, v, b.cls(v) orelse "?", t[c.i..]);
    }
    if (startsWith(u8, t, "[$")) {
        var c: Cur = .{ .s = t, .i = 2 };
        const v = c.word() orelse return error.Skip;
        // [$Rn] / [$Rn]!
        if (c.lit("]")) {
            const bang = c.lit("!");
            if (!c.end()) return error.Skip;
            if (!memBaseOk(b.cls(v))) return error.Skip;
            try b.setRename(v, "Rn");
            if (nxt) |n| if (n.len > 0 and n[0] == '$') {
                const pv = n[1..];
                const pc = b.cls(pv) orelse "";
                if (std.mem.indexOf(u8, b.rec.constraints, "wback") != null or startsWith(u8, pc, "GPR64pi")) {
                    b.consume_next = true;
                    if (startsWith(u8, pc, "GPR64pi")) {
                        try b.setRename(pv, "Rm");
                        b.alt_n = num(pc[7..]);
                        return one(a, .{ .k = "mem", .mode = "post", .mk = "postreg" });
                    }
                    if (eql(u8, pc, "simm9")) {
                        try b.setRename(pv, "offS");
                        return one(a, .{ .k = "mem", .mode = "post", .mk = "imm", .f = "offS", .scale = 0, .signed = true });
                    }
                    if (eql(u8, pc, "simm9s16")) {
                        try b.setRename(pv, "offS");
                        return one(a, .{ .k = "mem", .mode = "post", .mk = "imm", .f = "offS", .scale = 4, .signed = true });
                    }
                    if (eql(u8, pc, "simm7s4") or eql(u8, pc, "simm7s8") or eql(u8, pc, "simm7s16")) {
                        try b.setRename(pv, "offS");
                        return one(a, .{ .k = "mem", .mode = "post", .mk = "imm", .f = "offS", .scale = log2(num(pc[6..])), .signed = true });
                    }
                    return error.Skip;
                }
            };
            if (bang) return error.Skip;
            if (nxt) |n| if (n.len > 1 and n[0] == '#' and isNum(n[1..])) {
                b.consume_next = true;
                return one(a, .{ .k = "mem", .mode = "post", .mk = "fixed", .fixed = num(n[1..]) });
            };
            return one(a, .{ .k = "mem", .mode = "off", .mk = "base" });
        }
        if (!c.lit(", ")) return error.Skip;
        // [$Rn, #N] / [$Rn, #N]!
        if (c.lit("#")) {
            const st = c.i;
            _ = c.lit("-");
            if (c.digits() == null) return error.Skip;
            const nv = num(t[st..c.i]);
            if (!c.lit("]")) return error.Skip;
            const bang = c.lit("!");
            if (!c.end()) return error.Skip;
            if (!memBaseOk(b.cls(v))) return error.Skip;
            try b.setRename(v, "Rn");
            if (!bang) {
                if (nv != 0) return error.Skip;
                return one(a, .{ .k = "mem", .mode = "off", .mk = "base" });
            }
            return one(a, .{ .k = "mem", .mode = "pre", .mk = "fixed", .fixed = nv });
        }
        if (!c.lit("$")) return error.Skip;
        const o = c.word() orelse return error.Skip;
        if (c.lit("]")) {
            // [$Rn, $off] / [$Rn, $off]!
            const bang = c.lit("!");
            if (!c.end()) return error.Skip;
            if (!memBaseOk(b.cls(v))) return error.Skip;
            try b.setRename(v, "Rn");
            const mode: []const u8 = if (bang) "pre" else "off";
            const oc = b.cls(o) orelse "";
            {
                var p: Cur = .{ .s = oc };
                if (p.lit("uimm12s")) if (p.digits()) |d| if (p.end() and !bang) {
                    try b.setRename(o, "offZ");
                    return one(a, .{ .k = "mem", .mode = "off", .mk = "imm", .f = "offZ", .scale = log2(num(d)), .signed = false });
                };
            }
            {
                var p: Cur = .{ .s = oc };
                if (p.lit("simm9") or p.lit("simm7")) {
                    var sc: i64 = 1;
                    var ok = p.end();
                    if (!ok and p.lit("s")) if (p.digits()) |d| if (p.end()) {
                        sc = num(d);
                        ok = true;
                    };
                    if (ok) {
                        try b.setRename(o, "offS");
                        return one(a, .{ .k = "mem", .mode = mode, .mk = "imm", .f = "offS", .scale = log2(sc), .signed = true });
                    }
                }
            }
            if (eql(u8, oc, "simm10Scaled")) {
                try b.setRename(o, "offS");
                return one(a, .{ .k = "mem", .mode = mode, .mk = "imm", .f = "offS", .scale = 3, .signed = true });
            }
            return error.Skip;
        }
        // [$Rn, $Rm, $extend]
        if (!c.lit(", $")) return error.Skip;
        const ex = c.word() orelse return error.Skip;
        if (!c.lit("]") or !c.end()) return error.Skip;
        if (!eql(u8, b.cls(v) orelse "", "GPR64sp")) return error.Skip;
        var p: Cur = .{ .s = b.cls(ex) orelse "" };
        if (!p.lit("ro_")) return error.Skip;
        const wx = p.oneOf("WX") orelse return error.Skip;
        if (!p.lit("extend")) return error.Skip;
        const d = p.digits() orelse return error.Skip;
        if (!p.end()) return error.Skip;
        // roW is the same encoding as roX with a W index: one form (option + S) covers both
        if (wx == 'W') return error.Skip;
        try b.setRename(v, "Rn");
        try b.setRename(o, "Rm");
        const size = log2(@divTrunc(num(d), 8));
        try b.addOver("option", 13, 3, 0);
        try b.addOver("s", 12, 1, 0);
        try b.setSplit(ex, &.{.{ .lo = 0, .hi = 1, .name = "#0", .base = 0 }});
        if (size == 0) try b.setImm("ImmLDRB", &.{ -3, -3 });
        return one(a, .{ .k = "mem", .mode = "off", .mk = "reg", .scale = if (size > 0) size else -1 });
    }
    return error.Skip;
}

fn isGpr64Sp(c: []const u8) bool {
    return eql(u8, c, "GPR64sp") or eql(u8, c, "GPR64sponly");
}

/// One `$var` operand of class `c`, with the asm text that follows it (`.4s`, `$idx`, `/m`, ...).
///
/// LLVM operand class -> form operand (a64.zig `K`) and immediate encoding:
///   GPR32/GPR64 (+z, as32/as64, arg, noip)    w / x
///   GPR32sp/GPR64sp(+0, sponly)                wsp / xsp (SP allowed)
///   ... followed by arith/logical shift         + shift (shift3: lsl/lsr/asr, shift4: + ror)
///   ... followed by arith_extend*               rext/w + shift extend (option, imm3)
///   FPR8..FPR128                               sb..sq (scalar)
///   V64/V128 + `.4s`                           vfix (fixed arrangement); `.s$idx`: velem
///   VecList{One..Four}{arr} / {et}             vlist / vlistelem
///   ZPR*, Z_*                                  zfix; `$idx`: zelem
///   PPR8..PPR64, PPRorPNR*                     pfix;  PPR*Any / PPR_3b: pg, pgm (/m), pgz (/z)
///   arith/logical_shifted_reg*, arith_extended_reg*   register + shift operand (complex operand)
///   ccode / inv_ccode                          cond (inv_ccode: .inv_cond)
///   am_b*_target, am_brcond, am_tbrcond, adr(p)label   rel (scale 2, 0, 12)
///   am_ldrlit                                  memlit
///   vecshift{L,R}{8..64}[Narrow]               imm: ASimdSHL / ASimdShiftNImm (immh:immb);
///                                              SVE: SveShiftImm7 / 6 (tsz:imm3)
///   fixedpoint_f*_i*                           imm fbits: ASimdFBitsScaleImm (scale = 64 - fbits)
///   fpimm16/32/64, fpimm8                      fimm: ASimdFMovImm / VecVFPImm (SVE)
///   imm0_255/simdimmtype10 (movi, mvni, orr, bic)   imm + lsl/msl: ASimdMovPImm / MovNImm / LogicalImm
///   addsub_imm8_opt_lsl_i*, cpy_imm8_opt_lsl_i*     imm + lsl #8: Sve8BitImm / signed immS
///   sve_[preferred_]logical_imm*, logical_imm*  imm: SveLogicalImm / SveDupmImm / LogicalImm
///   addsub_shifted_imm32/64                    imm + lsl #12
///   movimm32/64_imm + shift                    imm + lsl: ImmWide (movz, movk) / ImmWideInv (movn)
///   t?imm*_0_N, uimmN, simmN_*b, i32imm, ...    imm / immS as-is;  uimmNsM: imm with scale M
///   X/WSeqPairClassOperand, GPR64x8            glist (2 or 8 consecutive registers)
///   msr/mrs_sysreg_op                          imm sysreg: ImmSysReg
///   prfop, rprfop                              imm: ImmPRF
///   sve_pred_enum, sve_incdec_imm              imm pattern (default 31): SvePatternImm; imm (default 1)
///   complexrotateop(odd)                       imm rotate/rot2: ASimdRotateImm_* / Rot2Imm, Rot1Imm
///   tbz_imm32_63                               imm (bit 5 -> Inst 31)
///   sve_fpimm_half_one/half_two/zero_one       fimm: FpHalfOrOneImm / FpHalfOrTwoImm / FpZeroOrOneImm
///   barrier_op                                 imm: ImmDataBarrier; isb: ImmISB (default 15)
///   sys_cr_op                                  imm (CRn/CRm)
/// Anything else is skipped (the record gives no form).
fn single(b: *B, v: []const u8, c: []const u8, rest: []const u8) Error![]Op {
    const a = b.a;
    if (lookup(&gpr_classes, c)) |k| {
        if (rest.len == 0) {
            const n = b.fname(v) orelse return error.Skip;
            return one(a, .{ .k = k, .f = n });
        }
        const sv = dollarWord(rest) orelse return error.Skip;
        const sc = b.cls(sv) orelse "";
        if (eql(u8, sc, "arith_shift32") or eql(u8, sc, "arith_shift64") or eql(u8, sc, "logical_shift32") or eql(u8, sc, "logical_shift64")) {
            try b.setSplit(sv, &.{ .{ .lo = 0, .hi = 5, .name = "n", .base = 0 }, .{ .lo = 6, .hi = 7, .name = "sop", .base = 0 } });
            return two(a, .{ .k = k, .f = b.fname(v) }, .{ .k = "shift", .sh = if (startsWith(u8, sc, "arith")) "shift3" else "shift4", .f = "n", .opt = true });
        }
        if (eql(u8, sc, "arith_extend") or eql(u8, sc, "arith_extend64") or eql(u8, sc, "arith_extendlsl64")) {
            try b.setSplit(sv, &.{ .{ .lo = 0, .hi = 2, .name = "n", .base = 0 }, .{ .lo = 3, .hi = 5, .name = "option", .base = 0 } });
            var wide = eql(u8, k, "x") or !eql(u8, sc, "arith_extend");
            for (b.rec.om.keys(), b.rec.om.values()) |x, xc| if (!eql(u8, x, v) and isGpr64Sp(xc)) {
                wide = true;
            };
            return two(a, .{ .k = if (wide) "rext" else k, .f = b.fname(v) }, .{ .k = "shift", .sh = "extend", .f = "n", .opt = true });
        }
        if (eql(u8, sc, "lsl_imm3_shift_operand")) {
            try b.setRename(sv, "n");
            return two(a, .{ .k = k, .f = b.fname(v) }, .{ .k = "shift", .sh = "lsl", .f = "n", .opt = true });
        }
        return error.Skip;
    }
    if (fprKind(c)) |k| if (rest.len == 0) return one(a, .{ .k = k, .f = b.fname(v) orelse return error.Skip });
    if (inList(&vreg_classes, c)) {
        var p: Cur = .{ .s = rest };
        if (p.lit(".")) {
            const st = p.i;
            const d = p.digits();
            if (d != null) if (p.oneOf("bhsdq") != null and p.end()) {
                return one(a, .{ .k = "vfix", .f = b.fname(v), .arr = lookup(&arrs, rest[st..]) orelse return error.Skip });
            };
            p.i = st + (if (d) |x| x.len else 0);
            const e = p.oneOf("bhsd") orelse return error.Skip;
            if (!p.lit("$")) return error.Skip;
            const iv = p.word() orelse return error.Skip;
            if (!p.end()) return error.Skip;
            var o: Op = .{ .k = "velem", .f = b.fname(v), .et = rest[p.i - iv.len - 2 .. p.i - iv.len - 1] };
            _ = e;
            if (d) |x| {
                const full = try std.mem.concat(a, u8, &.{ x, o.et.? });
                o.arr = lookup(&arrs, full) orelse return error.Skip;
            }
            const ic = b.cls(iv) orelse "";
            if (b.vs.contains(iv)) {
                const gop = try b.rename.getOrPut(a, iv);
                if (!gop.found_existing) gop.value_ptr.* = iv;
                o.f2 = gop.value_ptr.*;
            } else if (eql(u8, ic, "VectorIndex1")) {
                o.fixed = 1;
            } else if (eql(u8, ic, "VectorIndex0")) {
                o.fixed = 0;
            } else return error.Skip;
            return one(a, o);
        }
        return error.Skip;
    }
    if (startsWith(u8, c, "VecList")) {
        var p: Cur = .{ .s = c, .i = 7 };
        for (list_counts) |lc| if (p.lit(lc[0])) {
            const st = p.i;
            if (p.digits() != null and p.oneOf("bhsdq") != null and p.end()) {
                if (rest.len != 0) return error.Skip;
                return one(a, .{ .k = "vlist", .f = b.fname(v), .n = lc[1], .arr = lookup(&arrs, c[st..]) orelse return error.Skip });
            }
            p.i = st;
            if (p.oneOf("bhsd")) |_| if (p.end()) {
                const iv = dollarWord(rest) orelse return error.Skip;
                if (!b.vs.contains(iv)) return error.Skip;
                return one(a, .{ .k = "vlistelem", .f = b.fname(v), .n = lc[1], .et = c[st .. st + 1], .f2 = iv });
            };
            break;
        };
    }
    if (lookup(&zpr_classes, c)) |e| {
        if (rest.len == 0) return one(a, .{ .k = "zfix", .f = b.fname(v), .et = e });
        const iv = dollarWord(rest) orelse return error.Skip;
        if (b.vs.contains(iv)) return one(a, .{ .k = "zelem", .f = b.fname(v), .et = e, .f2 = iv });
        return error.Skip;
    }
    if (lookup(&ppr_classes, c)) |e| if (rest.len == 0) return one(a, .{ .k = "pfix", .f = b.fname(v), .et = e });
    if (inList(&pg_classes, c)) {
        const k: []const u8 = if (rest.len == 0) "pg" else if (eql(u8, rest, "/m")) "pgm" else if (eql(u8, rest, "/z")) "pgz" else return error.Skip;
        return one(a, .{ .k = k, .f = b.fname(v) });
    }
    // complex operands (register + shift/extend as one operand)
    if (eql(u8, c, "arith_shifted_reg32") or eql(u8, c, "arith_shifted_reg64") or eql(u8, c, "logical_shifted_reg32") or eql(u8, c, "logical_shifted_reg64")) {
        try b.setSplit("shift", &.{ .{ .lo = 0, .hi = 5, .name = "n", .base = 0 }, .{ .lo = 6, .hi = 7, .name = "sop", .base = 0 } });
        return two(a, .{ .k = if (endsWith(u8, c, "32")) "w" else "x", .f = "Rm" }, .{ .k = "shift", .sh = if (startsWith(u8, c, "arith")) "shift3" else "shift4", .f = "n", .opt = true });
    }
    if (eql(u8, c, "arith_extended_reg32_i32") or eql(u8, c, "arith_extended_reg32_i64") or eql(u8, c, "arith_extended_reg32to64_i64")) {
        try b.setSplit("extend", &.{ .{ .lo = 0, .hi = 2, .name = "n", .base = 0 }, .{ .lo = 3, .hi = 5, .name = "option", .base = 0 } });
        return two(a, .{ .k = if (eql(u8, c, "arith_extended_reg32_i32")) "w" else "rext", .f = "Rm" }, .{ .k = "shift", .sh = "extend", .f = "n", .opt = true });
    }
    if (eql(u8, c, "ccode")) return one(a, .{ .k = "cond", .f = v });
    if (eql(u8, c, "am_brcond") or eql(u8, c, "am_tbrcond") or eql(u8, c, "am_b_target") or eql(u8, c, "am_bl_target") or eql(u8, c, "am_brcmpcond")) {
        try b.setRename(v, "relS");
        return one(a, .{ .k = "rel", .f = "relS", .scale = 2 });
    }
    if (eql(u8, c, "adrlabel") or eql(u8, c, "adrplabel")) {
        try b.setRename(v, "relS");
        return one(a, .{ .k = "rel", .f = "relS", .scale = if (eql(u8, c, "adrlabel")) 0 else 12 });
    }
    if (eql(u8, c, "am_ldrlit")) {
        try b.setRename(v, "offS");
        return one(a, .{ .k = "memlit", .mode = "off", .mk = "imm", .f = "offS", .scale = 2, .signed = true });
    }
    {
        var p: Cur = .{ .s = c };
        if (p.lit("tvecshift") or p.lit("vecshift")) if (p.oneOf("LR")) |lr| if (p.digits()) |bits| {
            const narrow_cls = p.lit("Narrow");
            if (p.end() and rest.len == 0) {
                const left = lr == 'L';
                if (b.sve()) {
                    const mnm = try b.asmMnemonic();
                    const pos = b.vs.get(v) orelse return error.Skip;
                    var lows: u32 = 32;
                    for (pos.items) |pi| lows = @min(lows, pi[0]);
                    const narrow = endsWith(u8, mnm, "nb") or endsWith(u8, mnm, "nt") or endsWith(u8, mnm, "llb") or endsWith(u8, mnm, "llt");
                    try b.setSplit(v, &.{.{ .lo = 0, .hi = 6, .name = "#0", .base = 0 }});
                    try b.addOver("imm", 22, if (narrow) 1 else 2, 5);
                    try b.addOver("imm", lows, 5, 0);
                    try b.setImm(if (narrow) "SveShiftImm6" else "SveShiftImm7", &.{ -2, -3, if (left) 1 else 0 });
                    return one(a, .{ .k = "imm", .f = "imm" });
                }
                var l = log2(@divTrunc(num(bits), 8));
                if (narrow_cls) l -= 1;
                try b.addOver("immh", 19, 4, 0);
                try b.addOver("immb", 16, 3, 0);
                try b.setSplit(v, &.{.{ .lo = 0, .hi = 6, .name = "#0", .base = 0 }});
                try b.setImm(if (left) "ASimdSHL" else "ASimdShiftNImm", &.{ -3, l });
                return one(a, .{ .k = "imm", .f = "n" });
            }
        };
    }
    {
        var p: Cur = .{ .s = c };
        if (p.lit("fixedpoint_")) {
            _ = p.lit("recip_");
            if (p.lit("f") and p.digits() != null and p.lit("_i")) if (p.digits()) |iw| if (p.end() and rest.len == 0) {
                try b.addOver("scale", 10, 6, 0);
                try b.setSplit(v, &.{.{ .lo = 0, .hi = 6, .name = "#0", .base = 0 }});
                try b.setImm("ASimdFBitsScaleImm", &.{ -3, num(iw) });
                return one(a, .{ .k = "imm", .f = "fbits" });
            };
        }
    }
    if ((eql(u8, c, "fpimm16") or eql(u8, c, "fpimm32") or eql(u8, c, "fpimm64")) and rest.len == 0) {
        try b.setRename(v, "imm");
        try b.setImm(if (b.sve()) "VecVFPImm" else "ASimdFMovImm", &.{-3});
        return one(a, .{ .k = "fimm", .f = "fimm" });
    }
    if (eql(u8, c, "fpimm8") and rest.len == 0) {
        try b.setSplit(v, &.{ .{ .lo = 0, .hi = 4, .name = "defgh", .base = 0 }, .{ .lo = 5, .hi = 7, .name = "abc", .base = 0 } });
        try b.setImm("ASimdFMovImm", &.{-3});
        return one(a, .{ .k = "fimm", .f = "fimm" });
    }
    if (eql(u8, c, "imm0_255") or eql(u8, c, "simdimmtype10")) {
        const s = b.rec.asm_str;
        const t = std.mem.indexOfScalar(u8, s, '\t') orelse s.len;
        const mnm = s[0 .. std.mem.indexOfScalar(u8, s[0..t], '{') orelse t];
        if (inList(&.{ "movi", "mvni", "orr", "bic" }, mnm)) {
            const g2: ?[]const u8 = if (rest.len == 0) null else (dollarWord(rest) orelse return error.Skip);
            try b.setSplit(v, &.{ .{ .lo = 0, .hi = 4, .name = "defgh", .base = 0 }, .{ .lo = 5, .hi = 7, .name = "abc", .base = 0 } });
            if (g2) |x| try b.setSplit(x, &.{.{ .lo = 0, .hi = 7, .name = "#0", .base = 0 }});
            try b.addOver("cmode", 12, 4, 0);
            const mov = eql(u8, mnm, "movi") or eql(u8, mnm, "mvni");
            if (mov) try b.addOver("op", 29, 1, 0);
            try b.setImm(if (eql(u8, mnm, "movi")) "ASimdMovPImm" else if (eql(u8, mnm, "mvni")) "ASimdMovNImm" else "ASimdLogicalImm", &.{-3});
            return two(a, .{ .k = "imm", .f = "imm" }, .{ .k = "shift", .sh = if (mov) "lslmsl" else "lsl", .f = "n", .opt = true });
        }
    }
    {
        var p: Cur = .{ .s = c };
        const cpy = p.lit("cpy");
        if ((cpy or p.lit("addsub")) and p.lit("_imm8_opt_lsl_i") and p.digits() != null and p.end() and rest.len == 0) {
            if (cpy) {
                try b.setSplit(v, &.{ .{ .lo = 0, .hi = 7, .name = "immS", .base = 0 }, .{ .lo = 8, .hi = 8, .name = "n", .base = 0 } });
                return two(a, .{ .k = "imm", .f = "immS" }, .{ .k = "shift", .sh = "lsl", .f = "n", .opt = true, .scale = 8 });
            }
            try b.setSplit(v, &.{ .{ .lo = 0, .hi = 7, .name = "imm", .base = 0 }, .{ .lo = 8, .hi = 8, .name = "n", .base = 0 } });
            try b.setImm("Sve8BitImm", &.{-3});
            return two(a, .{ .k = "imm", .f = "imm" }, .{ .k = "shift", .sh = "lsl", .f = "n", .opt = true, .scale = 8 });
        }
    }
    {
        var p: Cur = .{ .s = c };
        var sve_log = false;
        if (p.lit("sve_")) {
            _ = p.lit("preferred_");
            if (p.lit("logical_imm") and p.digits() != null) {
                _ = p.lit("_not");
                sve_log = p.end();
            }
        }
        if ((sve_log or ((eql(u8, c, "logical_imm64") or eql(u8, c, "logical_imm64_not")) and b.sve())) and rest.len == 0) {
            try b.setRename(v, "imm");
            const mnm = try b.asmMnemonic();
            if (eql(u8, mnm, "dupm") or eql(u8, mnm, "mov")) {
                try b.setImm("SveDupmImm", &.{ -2, -3 });
            } else try b.setImm("SveLogicalImm", &.{ -3, if (endsWith(u8, c, "_not")) 1 else 0, -3 });
            return one(a, .{ .k = "imm", .f = "imm" });
        }
    }
    if (plainImm(c)) |nm0| if (rest.len == 0) {
        const nm = if (eql(u8, nm0, "immS")) nm0 else b.ren(v);
        try b.setRename(v, nm);
        return one(a, .{ .k = "imm", .f = nm });
    };
    {
        var p: Cur = .{ .s = c };
        if (p.lit("uimm") and p.digits() != null and p.lit("s")) if (p.digits()) |sc| if (p.end() and rest.len == 0) {
            return one(a, .{ .k = "imm", .f = b.ren(v), .scale = log2(num(sc)) });
        };
    }
    if (eql(u8, c, "movimm32_imm") or eql(u8, c, "movimm64_imm") or eql(u8, c, "imm0_65535")) if (dollarWord(rest)) |sv| {
        try b.setRename(v, "imm");
        try b.setSplit(sv, &.{ .{ .lo = 0, .hi = 3, .name = "#0", .base = 0 }, .{ .lo = 4, .hi = 5, .name = "hw", .base = 0 } });
        const s = b.rec.asm_str;
        const mnm = s[0 .. std.mem.indexOfScalar(u8, s, '\t') orelse s.len];
        var wide: i64 = 0;
        for (b.rec.om.values()) |xc| if (eql(u8, xc, "GPR64")) {
            wide = 1;
        };
        if (eql(u8, mnm, "movz") or eql(u8, mnm, "movk")) try b.setImm("ImmWide", &.{ -3, -3, wide }) else if (eql(u8, mnm, "movn")) try b.setImm("ImmWideInv", &.{ -3, -3, wide });
        return two(a, .{ .k = "imm", .f = "imm" }, .{ .k = "shift", .sh = "lsl", .f = "n", .opt = true });
    };
    if ((eql(u8, c, "XSeqPairClassOperand") or eql(u8, c, "WSeqPairClassOperand")) and rest.len == 0)
        return one(a, .{ .k = "glist", .f = b.fname(v), .n = 2, .w = c[0] == 'X' });
    if (eql(u8, c, "GPR64x8") and rest.len == 0) return one(a, .{ .k = "glist", .f = b.fname(v), .n = 8, .w = true });
    if ((eql(u8, c, "msr_sysreg_op") or eql(u8, c, "mrs_sysreg_op")) and rest.len == 0) {
        try b.addOver("sysreg", 5, 15, 0);
        try b.forced.put(a, 20, 1);
        try b.setSplit(v, &.{.{ .lo = 0, .hi = 15, .name = "#0", .base = 0 }});
        try b.setImm("ImmSysReg", &.{-3});
        return one(a, .{ .k = "imm", .f = "sysreg" });
    }
    if ((eql(u8, c, "prfop") or eql(u8, c, "rprfop")) and rest.len == 0) {
        try b.setImm("ImmPRF", &.{-3});
        return one(a, .{ .k = "imm", .f = b.fname(v) });
    }
    if (eql(u8, c, "sve_pred_enum") and rest.len == 0) {
        try b.setRename(v, "pattern");
        try b.setImm("SvePatternImm", &.{-3});
        return one(a, .{ .k = "imm", .f = "pattern", .opt = true, .fixed = 31 });
    }
    if (eql(u8, c, "sve_incdec_imm") and rest.len == 0) {
        try b.setRename(v, "imm");
        return one(a, .{ .k = "imm", .f = "imm", .opt = true, .fixed = 1 });
    }
    if ((eql(u8, c, "complexrotateop") or eql(u8, c, "complexrotateopodd")) and rest.len == 0) {
        try b.setRename(v, "imm");
        const sve = b.sve();
        if (eql(u8, c, "complexrotateop")) try b.setImm(if (sve) "Rot2Imm" else "ASimdRotateImm_0_90_180_270", &.{-3}) else try b.setImm(if (sve) "Rot1Imm" else "ASimdRotateImm_90_270", &.{-3});
        return one(a, .{ .k = "imm", .f = if (sve) "rot2" else "rotate" });
    }
    if (eql(u8, c, "tbz_imm32_63") and rest.len == 0) {
        try b.setSplit(v, &.{.{ .lo = 0, .hi = 4, .name = "imm", .base = 0 }});
        try b.addOver("imm", 31, 1, 5);
        return one(a, .{ .k = "imm", .f = "imm" });
    }
    if (rest.len == 0) {
        const fp = [_]KV{ .{ "sve_fpimm_half_one", "FpHalfOrOneImm" }, .{ "sve_fpimm_half_two", "FpHalfOrTwoImm" }, .{ "sve_fpimm_zero_one", "FpZeroOrOneImm" } };
        if (lookup(&fp, c)) |fnm| {
            try b.setRename(v, "imm");
            try b.setImm(fnm, &.{-3});
            return one(a, .{ .k = "fimm", .f = "fimm" });
        }
        if (eql(u8, c, "logical_imm32") or eql(u8, c, "logical_imm64")) {
            try b.setRename(v, "imm");
            try b.setImm("LogicalImm", &.{ -3, if (eql(u8, c, "logical_imm64")) 1 else 0 });
            return one(a, .{ .k = "imm", .f = "imm" });
        }
        if (eql(u8, c, "addsub_shifted_imm32") or eql(u8, c, "addsub_shifted_imm64")) {
            try b.setSplit(v, &.{ .{ .lo = 0, .hi = 11, .name = "imm", .base = 0 }, .{ .lo = 12, .hi = 12, .name = "n", .base = 0 }, .{ .lo = 13, .hi = 13, .name = "#0", .base = 0 } });
            return two(a, .{ .k = "imm", .f = "imm" }, .{ .k = "shift", .sh = "lsl", .f = "n", .opt = true, .scale = 12 });
        }
        if (eql(u8, c, "barrier_op")) {
            if (eql(u8, try b.asmMnemonic(), "isb")) {
                try b.setImm("ImmISB", &.{-3});
                return one(a, .{ .k = "imm", .f = "isb_op", .opt = true, .fixed = 15 });
            }
            try b.setImm("ImmDataBarrier", &.{-3});
            return one(a, .{ .k = "imm", .f = "barrier_op" });
        }
        if (eql(u8, c, "sys_cr_op")) {
            const nm: []const u8 = if (eql(u8, v, "Cn")) "CRn" else if (eql(u8, v, "Cm")) "CRm" else v;
            try b.setRename(v, nm);
            return one(a, .{ .k = "imm", .f = nm });
        }
        if (eql(u8, c, "inv_ccode")) {
            b.inv_cond = true;
            return one(a, .{ .k = "cond", .f = v });
        }
    }
    return error.Skip;
}

/// Split the operand part of an asm string at top-level commas.
fn splitOps(a: Allocator, s: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var depth: i32 = 0;
    var st: usize = 0;
    for (s, 0..) |ch, i| {
        if (ch == '[' or ch == '{') depth += 1;
        if (ch == ']' or ch == '}') depth -= 1;
        if (ch == ',' and depth == 0) {
            try out.append(a, std.mem.trim(u8, s[st..i], " \t"));
            st = i + 1;
        }
    }
    const last = std.mem.trim(u8, s[st..], " \t");
    if (last.len > 0) try out.append(a, last);
    return out.items;
}

fn tiesOf(a: Allocator, b: *B, constraints: []const u8) !void {
    var it = std.mem.splitScalar(u8, constraints, ',');
    while (it.next()) |part| {
        var c: Cur = .{ .s = std.mem.trimStart(u8, part, " \t") };
        if (!c.lit("$")) continue;
        const x = c.word() orelse continue;
        while (c.lit(" ")) {}
        if (!c.lit("=")) continue;
        while (c.lit(" ")) {}
        if (!c.lit("$")) continue;
        const y = c.word() orelse continue;
        try b.tie.put(a, x, y);
        try b.tie.put(a, y, x);
    }
}

/// One record (or alias rewritten onto its target) -> forms (two for a post-index register form:
/// `[Xn], Xm` and the fixed `[Xn], #imm` twin).
fn build(a: Allocator, name: []const u8, rec: *const Rec) Error![]Form {
    var b: B = .{ .a = a, .rec = rec, .name = name };
    for (rec.inst, 0..) |bit, i| switch (bit) {
        .fixed => |x| b.fixed_val |= @as(u32, x) << @intCast(i),
        .v => |v| {
            const gop = try b.vs.getOrPut(a, v.name);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(a, .{ @intCast(i), v.idx });
        },
        .none => {},
    };
    try tiesOf(a, &b, rec.constraints);
    const asm_text = std.mem.trim(u8, try lj.asmVariant(a, rec.asm_str, 0), " \t\n");
    var mn_end: usize = 0;
    while (mn_end < asm_text.len and !std.ascii.isWhitespace(asm_text[mn_end])) mn_end += 1;
    var mn = asm_text[0..mn_end];
    var toks: std.ArrayList([]const u8) = .empty;
    if (std.mem.indexOf(u8, mn, ".$cond")) |p| {
        mn = try std.mem.concat(a, u8, &.{ mn[0..p], "_cond", mn[p + 6 ..] });
        try toks.append(a, "$cond");
    }
    if (std.mem.indexOfScalar(u8, mn, '$') != null) return error.Skip;
    try toks.appendSlice(a, try splitOps(a, std.mem.trimStart(u8, asm_text[mn_end..], " \t")));
    var ops: std.ArrayList(Op) = .empty;
    var i: usize = 0;
    while (i < toks.items.len) {
        const t = toks.items[i];
        i += 1;
        const nxt: ?[]const u8 = if (i < toks.items.len) toks.items[i] else null;
        try ops.appendSlice(a, try token(&b, t, nxt));
        if (b.consume_next) {
            b.consume_next = false;
            i += 1;
        }
    }
    // fields from the Inst layout
    var over_pos: u32 = 0;
    for (b.over.items) |o| over_pos |= @truncate(((@as(u64, 1) << @intCast(o.w)) - 1) << @intCast(o.at));
    var bits = b.fixed_val;
    for (b.forced.keys(), b.forced.values()) |p, v| bits = (bits & ~(@as(u32, 1) << @intCast(p))) | (@as(u32, v) << @intCast(p));
    bits &= ~over_pos;
    var fl: std.ArrayList(Field) = .empty;
    for (b.vs.keys(), b.vs.values()) |v, list| for (list.items) |pi| {
        const pos = pi[0];
        const idx = pi[1];
        if ((over_pos >> @intCast(pos)) & 1 == 1 or b.forced.contains(pos)) continue;
        var nm: []const u8 = undefined;
        var src = idx;
        if (b.split.get(v)) |parts| {
            var found = false;
            for (parts) |s| if (s.lo <= idx and idx <= s.hi) {
                nm = s.name;
                src = idx - s.lo + s.base;
                found = true;
                break;
            };
            if (!found) return error.Skip;
            if (eql(u8, nm, "#0")) continue;
            if (eql(u8, nm, "#1")) {
                bits |= @as(u32, 1) << @intCast(pos);
                continue;
            }
        } else nm = b.ren(v);
        try fl.append(a, .{ .n = nm, .at = pos, .w = 1, .src = src });
    };
    try fl.appendSlice(a, b.over.items);
    std.mem.sort(Field, fl.items, {}, struct {
        fn lt(_: void, x: Field, y: Field) bool {
            return x.at < y.at;
        }
    }.lt);
    var merged: std.ArrayList(Field) = .empty;
    for (fl.items) |f| {
        if (merged.items.len > 0) {
            const l = &merged.items[merged.items.len - 1];
            if (eql(u8, l.n, f.n) and l.at + l.w == f.at and l.src + l.w == f.src) {
                l.w += f.w;
                continue;
            }
        }
        try merged.append(a, f);
    }
    std.mem.reverse(Field, merged.items);
    // every register operand needs its field
    for (ops.items) |o| for ([_]?[]const u8{ o.f, o.f2 }) |fo| if (fo) |fname| {
        var found = false;
        for (merged.items) |f| if (eql(u8, f.n, fname)) {
            found = true;
        };
        if (!found and !(o.opt orelse false) and !inList(&.{ "imm", "shift", "fimm", "cond", "rel" }, o.k)) return error.Skip;
    };
    const res: Form = .{ .mn = mn, .bits = bits, .fields = merged.items, .ops = ops.items, .imm = b.imm, .args = b.args, .rec = name, .inv_cond = b.inv_cond };
    if (b.alt_n) |n| {
        var alt = res;
        var af: std.ArrayList(Field) = .empty;
        var rm_at: u32 = 0;
        for (merged.items) |f| if (eql(u8, f.n, "Rm")) {
            rm_at = f.at;
        } else try af.append(a, f);
        alt.fields = af.items;
        alt.bits = bits | (@as(u32, 31) << @intCast(rm_at));
        const aops = try a.dupe(Op, ops.items);
        for (aops) |*o| if (o.mk) |mk| if (eql(u8, mk, "postreg")) {
            o.mk = "fixed";
            o.fixed = n;
        };
        alt.ops = aops;
        alt.rec = try std.mem.concat(a, u8, &.{ name, " (#imm)" });
        const r = try a.alloc(Form, 2);
        r[0] = alt;
        r[1] = res;
        return r;
    }
    const r = try a.alloc(Form, 1);
    r[0] = res;
    return r;
}

// ---- aliases ---------------------------------------------------------------------------------

fn hwenc(db: lj.Db, reg: []const u8) Skip!i64 {
    const r = db.get(reg) orelse return error.Skip;
    var v: i64 = 0;
    for (lj.list(r, "HWEncoding"), 0..) |x, i| if (x == .integer) {
        v |= x.integer << @intCast(i);
    };
    return v;
}

fn items(v: Value) Skip![]const Value {
    return if (v == .array) v.array.items else error.Skip;
}
fn pairName(p: []const Value) ?[]const u8 {
    return if (p.len > 1 and p[1] == .string) p[1].string else null;
}

fn isDag(v: Value) bool {
    return v == .object and eql(u8, lj.str(v.object, "kind"), "dag");
}

/// An `InstAlias` as a record of its target: the alias asm string over the target's `Inst`
/// layout, with the alias operand names substituted and constant operands folded into the bits.
fn aliasRecord(a: Allocator, db: lj.Db, al: Obj, tgt_out: *[]const u8) Error!Rec {
    const ri = (al.get("ResultInst") orelse return error.Skip);
    if (ri != .object) return error.Skip;
    const tgt = lj.def(ri.object, "operator");
    tgt_out.* = tgt;
    const r = db.get(tgt) orelse return error.Skip;
    if (lj.int(r, "isPseudo") != 0 or lj.int(r, "isCodeGenOnly") != 0) return error.Skip;
    const inst_v = r.get("Inst") orelse return error.Skip;
    if (inst_v != .array) return error.Skip;
    var tb: B = .{ .a = a, .rec = undefined, .name = "" };
    try tiesOf(a, &tb, lj.str(r, "Constraints"));
    const Slot = struct { nm: []const u8, sub: ?[]const []const u8 };
    var slots: std.ArrayList(Slot) = .empty;
    const outs = dagArgs(r, "OutOperandList");
    var outnames: std.StringHashMapUnmanaged(void) = .empty;
    for (outs) |o| try outnames.put(a, pairOf(o).name, {});
    for ([_][]const Value{ outs, dagArgs(r, "InOperandList") }, 0..) |lst, li| for (lst) |pv| {
        const p = pairOf(pv);
        if (li == 1) if (tb.tie.get(p.name)) |t| if (outnames.contains(t)) continue;
        var sub: ?[]const []const u8 = null;
        if (isDag(p.cls)) {
            var sl: std.ArrayList([]const u8) = .empty;
            for (try items(p.cls.object.get("args") orelse return error.Skip)) |x| try sl.append(a, pairName(try items(x)) orelse "");
            sub = sl.items;
        }
        try slots.append(a, .{ .nm = p.name, .sub = sub });
    };
    const args = try items(ri.object.get("args") orelse return error.Skip);
    if (args.len != slots.items.len) return error.Skip;
    var ren: std.StringHashMapUnmanaged([]const u8) = .empty;
    var cst: std.StringHashMapUnmanaged(i64) = .empty;
    var rec: Rec = .{ .asm_str = lj.str(al, "AsmString"), .inst = undefined, .constraints = "" };
    for (slots.items, args) |s, pv| {
        const pa = try items(pv);
        if (pa.len == 0) return error.Skip;
        const av = pa[0];
        const an = pairName(pa);
        switch (av) {
            .integer => |x| {
                try cst.put(a, s.nm, x);
                continue;
            },
            .array => |bl| {
                // bits<N> constant, least significant bit first
                var x: i64 = 0;
                for (bl.items, 0..) |bv, i| {
                    if (bv != .integer) return error.Skip;
                    x |= bv.integer << @intCast(i);
                }
                try cst.put(a, s.nm, x);
                continue;
            },
            else => {},
        }
        if (isDag(av)) {
            const sargs = try items(av.object.get("args") orelse return error.Skip);
            const sub = s.sub orelse {
                // complex operand spelled as constants, e.g. (cpy_imm8_opt_lsl_i16 0, 0)
                for (sargs) |x| {
                    const xi = try items(x);
                    if (xi.len == 0 or xi[0] != .integer or xi[0].integer != 0) return error.Skip;
                }
                try cst.put(a, s.nm, 0);
                continue;
            };
            for (sub, 0..) |sn, i| {
                if (i >= sargs.len) break;
                const si = try items(sargs[i]);
                if (si.len == 0) return error.Skip;
                const sv = si[0];
                const san = pairName(si);
                if (sv == .integer) {
                    try cst.put(a, sn, sv.integer);
                } else if (sv == .object and eql(u8, lj.str(sv.object, "kind"), "def") and san == null) {
                    try cst.put(a, sn, try hwenc(db, lj.str(sv.object, "def")));
                } else if (san) |x| {
                    try ren.put(a, sn, x);
                    try rec.om.put(a, x, if (sv == .object) lj.str(sv.object, "def") else "?");
                } else return error.Skip;
            }
            continue;
        }
        if (av == .object and eql(u8, lj.str(av.object, "kind"), "def") and an == null) {
            try cst.put(a, s.nm, try hwenc(db, lj.str(av.object, "def")));
            continue;
        }
        const x = an orelse return error.Skip;
        if (av != .object) return error.Skip;
        try ren.put(a, s.nm, x);
        try rec.om.put(a, x, lj.str(av.object, "def"));
        if (s.sub) |sub| for (sub) |sn| try ren.put(a, sn, x);
    }
    const inst = instBits(inst_v.array.items) orelse return error.Skip;
    for (inst, 0..) |bit, i| {
        rec.inst[i] = bit;
        if (bit != .v) continue;
        const v = bit.v.name;
        if (cst.get(v)) |x| {
            rec.inst[i] = .{ .fixed = @intCast((x >> @intCast(bit.v.idx)) & 1) };
        } else if (ren.get(v)) |x| {
            rec.inst[i] = .{ .v = .{ .name = x, .idx = bit.v.idx } };
        } else if (tb.tie.get(v)) |t| if (ren.get(t)) |x| {
            rec.inst[i] = .{ .v = .{ .name = x, .idx = bit.v.idx } };
        };
    }
    return rec;
}

// ---- project decisions (not LLVM data) -------------------------------------------------------

const FixedField = struct { []const u8, i64 };
const Manual = struct {
    mn: []const u8,
    rec: []const u8,
    ops: []const Op,
    imm: []const u8,
    args: []const i64,
    fixed: []const FixedField = &.{},
    /// bti: the hint immediate becomes CRm = 0b0100 and op2 = targets << 1
    bti: bool = false,
};

fn bf(comptime mn: []const u8, comptime rec: []const u8, comptime fnm: []const u8) [2]Manual {
    return .{
        .{ .mn = mn, .rec = rec ++ "Wri", .ops = &.{ .{ .k = "w", .f = "Rd" }, .{ .k = "w", .f = "Rn" }, .{ .k = "imm", .f = "lsb" }, .{ .k = "imm", .f = "width" } }, .imm = fnm, .args = &.{ -3, -3, 0 } },
        .{ .mn = mn, .rec = rec ++ "Xri", .ops = &.{ .{ .k = "x", .f = "Rd" }, .{ .k = "x", .f = "Rn" }, .{ .k = "imm", .f = "lsb" }, .{ .k = "imm", .f = "width" } }, .imm = fnm, .args = &.{ -3, -3, 1 } },
    };
}
fn sysAlias(comptime mn: []const u8, comptime fnm: []const u8, comptime crn: i64, comptime rt_optional: bool) []const Manual {
    const op: Op = .{ .k = "imm", .f = mn ++ "_op" };
    const with_rt: Manual = .{ .mn = mn, .rec = "SYSxt", .ops = &.{ op, .{ .k = "x", .f = "Rt" } }, .imm = fnm, .args = &.{-3}, .fixed = &.{.{ "CRn", crn }} };
    if (!rt_optional) return &.{with_rt};
    return &.{ .{ .mn = mn, .rec = "SYSxt", .ops = &.{op}, .imm = fnm, .args = &.{-3}, .fixed = &.{ .{ "CRn", crn }, .{ "Rt", 31 } } }, with_rt };
}

/// Aliases LLVM implements in C++ (AArch64AsmParser), not as InstAlias records, plus AsmJit
/// compatibility forms. Each names the LLVM record that supplies the bits and fields; only the
/// operand list, the immediate conversion and fixed fields are written here. They go first in
/// their mnemonic's list (LLVM's parser rewrites them before matching).
const manual = [_]Manual{
    // mov Rd, #imm -> movz / movn / orr Rd, zr, #bitmask (in this order, as llvm-mc)
    .{ .mn = "mov", .rec = "MOVZWi", .ops = &.{ .{ .k = "w", .f = "Rd" }, .{ .k = "imm", .f = "imm" } }, .imm = "ImmWide", .args = &.{ -3, 0, 0 } },
    .{ .mn = "mov", .rec = "MOVZXi", .ops = &.{ .{ .k = "x", .f = "Rd" }, .{ .k = "imm", .f = "imm" } }, .imm = "ImmWide", .args = &.{ -3, 0, 1 } },
    .{ .mn = "mov", .rec = "MOVNWi", .ops = &.{ .{ .k = "w", .f = "Rd" }, .{ .k = "imm", .f = "imm" } }, .imm = "ImmWideInv", .args = &.{ -3, 0, 0 } },
    .{ .mn = "mov", .rec = "MOVNXi", .ops = &.{ .{ .k = "x", .f = "Rd" }, .{ .k = "imm", .f = "imm" } }, .imm = "ImmWideInv", .args = &.{ -3, 0, 1 } },
    .{ .mn = "mov", .rec = "ORRWri", .ops = &.{ .{ .k = "wsp", .f = "Rd" }, .{ .k = "imm", .f = "imm" } }, .imm = "LogicalImm", .args = &.{ -3, 0 }, .fixed = &.{.{ "Rn", 31 }} },
    .{ .mn = "mov", .rec = "ORRXri", .ops = &.{ .{ .k = "xsp", .f = "Rd" }, .{ .k = "imm", .f = "imm" } }, .imm = "LogicalImm", .args = &.{ -3, 1 }, .fixed = &.{.{ "Rn", 31 }} },
    // lsl Rd, Rn, #n -> ubfm
    .{ .mn = "lsl", .rec = "UBFMWri", .ops = &.{ .{ .k = "w", .f = "Rd" }, .{ .k = "w", .f = "Rn" }, .{ .k = "imm", .f = "n" } }, .imm = "ImmLSL", .args = &.{ -3, 0 } },
    .{ .mn = "lsl", .rec = "UBFMXri", .ops = &.{ .{ .k = "x", .f = "Rd" }, .{ .k = "x", .f = "Rn" }, .{ .k = "imm", .f = "n" } }, .imm = "ImmLSL", .args = &.{ -3, 1 } },
} ++ bf("bfi", "BFM", "ImmBFIZ") ++ bf("bfxil", "BFM", "ImmBFX") ++ bf("sbfiz", "SBFM", "ImmBFIZ") ++ bf("sbfx", "SBFM", "ImmBFX") ++
    bf("ubfiz", "UBFM", "ImmBFIZ") ++ bf("ubfx", "UBFM", "ImmBFX") ++ [_]Manual{
    // bfc Rd, #lsb, #width -> bfm Rd, zr, ...
    .{ .mn = "bfc", .rec = "BFMWri", .ops = &.{ .{ .k = "w", .f = "Rd" }, .{ .k = "imm", .f = "lsb" }, .{ .k = "imm", .f = "width" } }, .imm = "ImmBFIZ", .args = &.{ -3, -3, 0 }, .fixed = &.{.{ "Rn", 31 }} },
    .{ .mn = "bfc", .rec = "BFMXri", .ops = &.{ .{ .k = "x", .f = "Rd" }, .{ .k = "imm", .f = "lsb" }, .{ .k = "imm", .f = "width" } }, .imm = "ImmBFIZ", .args = &.{ -3, -3, 1 }, .fixed = &.{.{ "Rn", 31 }} },
    // bti {#targets} -> hint #(32 | targets << 1)
    .{ .mn = "bti", .rec = "HINT", .ops = &.{.{ .k = "imm", .f = "targets", .opt = true, .fixed = 0 }}, .imm = "ImmBTI", .args = &.{-3}, .bti = true },
} ++
    // ic/dc/at/tlbi <op>{, Xt} -> sys; the operation is one immediate (op1 << 11 | CRn << 7 |
    // CRm << 3 | op2) and CRn is fixed per mnemonic
    sysAlias("ic", "ImmIC", 7, true)[0..2].* ++ sysAlias("dc", "ImmDC", 7, false)[0..1].* ++
    sysAlias("at", "ImmAt", 7, false)[0..1].* ++ sysAlias("tlbi", "ImmTLBI", 8, true)[0..2].* ++ [_]Manual{
    // AsmJit compatibility: sshll2/ushll2 Vd.2S, Vn.4S, #n encode as Vd.2D (the upstream AsmJit
    // tests spell it this way)
    .{ .mn = "sshll2", .rec = "SSHLLv4i32_shift", .ops = &.{ .{ .k = "vfix", .f = "Rd", .arr = "a2s" }, .{ .k = "vfix", .f = "Rn", .arr = "a4s" }, .{ .k = "imm", .f = "n" } }, .imm = "ASimdSHL", .args = &.{ -3, 2 } },
    .{ .mn = "ushll2", .rec = "USHLLv4i32_shift", .ops = &.{ .{ .k = "vfix", .f = "Rd", .arr = "a2s" }, .{ .k = "vfix", .f = "Rn", .arr = "a4s" }, .{ .k = "imm", .f = "n" } }, .imm = "ASimdSHL", .args = &.{ -3, 2 } },
};

/// sxtb/sxth/sxtw Xd, Wn: LLVM's parser accepts the W source of the X alias.
const src_w = [_][]const u8{ "sxtb", "sxth", "sxtw" };

/// ADD/SUB *Xrx64: the option field is partly fixed (UXTX/SXTX only); the *Xrx form, whose index
/// operand takes W or X (`rext`), covers these encodings.
fn skippedRecord(n: []const u8) bool {
    var c: Cur = .{ .s = n };
    if (!(c.lit("ADD") or c.lit("SUB"))) return false;
    _ = c.lit("S");
    return c.lit("Xrx64") and c.end();
}

/// Form order decides which encoding wins when several match; follow llvm-mc.
fn lowPriority(f: Form) bool {
    // `mov Zd, #imm`: DUP (integer) before DUPM (bitmask)
    if (std.mem.indexOf(u8, f.rec, "DUPM_ZI") != null) return true;
    // `mov Xd, Xn`: ORR; the ADD #0 alias is for SP
    if (eql(u8, f.mn, "mov") and (startsWith(u8, f.rec, "ADDWri") or startsWith(u8, f.rec, "ADDXri"))) return true;
    // `cmp w1, w2`: shifted register before extended register (the SP forms)
    var c: Cur = .{ .s = f.rec };
    if (c.lit("ADD") or c.lit("SUB")) {
        _ = c.lit("S");
        if (c.oneOf("WX") != null and c.lit("rx")) return true;
    }
    return false;
}

/// Field names a64.zig refers to that no generated form uses.
const extra_names = [_][]const u8{ "!post", "M", "W", "bits", "log_imm", "rot1", "sf" };

/// Immediate conversions a64.zig implements (`ImmFn`).
const imm_fns = [_][]const u8{
    "ASimdFBitsHBImm", "ASimdFBitsScaleImm",          "ASimdFMovImm",          "ASimdLogicalImm", "ASimdMovNImm",
    "ASimdMovPImm",    "ASimdRotateImm_0_90_180_270", "ASimdRotateImm_90_270", "ASimdSHL",        "ASimdSHRN",
    "ASimdShiftNImm",  "ASimdShiftPImm",              "ASimdXtlImm",           "FpHalfOrOneImm",  "FpHalfOrTwoImm",
    "FpZeroOrOneImm",  "ImmAt",                       "ImmBFIZ",               "ImmBFM",          "ImmBFX",
    "ImmBRB",          "ImmBTI",                      "ImmDC",                 "ImmDataBarrier",  "ImmIC",
    "ImmISB",          "ImmLDR",                      "ImmLDRB",               "ImmLDRB_STRB",    "ImmLDRH",
    "ImmLDRH_STRH",    "ImmLDRW",                     "ImmLDR_STR",            "ImmLSL",          "ImmLogical",
    "ImmMSR",          "ImmPRF",                      "ImmPSB_TSB",            "ImmRCTX",         "ImmRPRF",
    "ImmSMEMode",      "ImmSysReg",                   "ImmTLBI",               "ImmTLBIP",        "ImmWide",
    "ImmWideInv",      "LogicalImm",                  "Rot1Imm",               "Rot2Imm",         "ShiftImm",
    "Sve8BitImm",      "SveDupImm",                   "SveDupmImm",            "SveLogicalImm",   "SvePatternImm",
    "SveShiftImm6",    "SveShiftImm7",                "SveXtImm",              "VecVFPImm",       "none",
};

/// AsmJit mnemonics with no LLVM form here, kept in `Mnemonic` so existing code compiles (encoding
/// them returns NoMatchingForm). Either not in the architecture (`stadda`, the TME `tstart`
/// family dropped from LLVM, `fcvtx2`) or spelled with a keyword operand (`gcsb dsync`,
/// `psb csync`, `smstart sm`, `cfp rctx, Xt`) that this API has no operand for.
const kept_mnemonics = [_][]const u8{
    "brb",       "tlbip",    "gcsb",      "ld64b",    "st64b",     "st64bv0",  "st64bv",    "stadda",   "staddal",   "staddab",
    "staddalb",  "staddah",  "staddalh",  "stclra",   "stclral",   "stclrab",  "stclralb",  "stclrah",  "stclralh",  "steora",
    "steoral",   "steorab",  "steoralb",  "steorah",  "steoralh",  "stseta",   "stsetal",   "stsetab",  "stsetalb",  "stsetah",
    "stsetalh",  "stsmaxa",  "stsmaxal",  "stsmaxab", "stsmaxalb", "stsmaxah", "stsmaxalh", "stsmina",  "stsminal",  "stsminab",
    "stsminalb", "stsminah", "stsminalh", "stumaxa",  "stumaxal",  "stumaxab", "stumaxalb", "stumaxah", "stumaxalh", "stumina",
    "stuminal",  "stuminab", "stuminalb", "stuminah", "stuminalh", "psb",      "cfp",       "cpp",      "dvp",       "cosp",
    "mrrs",      "msrr",     "tcancel",   "tcommit",  "tstart",    "ttest",    "tsb",       "fcvtx2",   "smstart",   "smstop",
};

fn manualForm(a: Allocator, db: lj.Db, m: Manual) Error!Form {
    const r = db.get(m.rec) orelse return error.Skip;
    const rec = (try recordOf(a, r)) orelse return error.Skip;
    const base = (try build(a, m.rec, &rec))[0];
    var bits = base.bits;
    var fields: std.ArrayList(Field) = .empty;
    for (base.fields) |f| {
        if (m.bti and eql(u8, f.n, "imm")) continue;
        var drop = false;
        for (m.fixed) |fx| if (eql(u8, fx[0], f.n)) {
            bits |= @as(u32, @intCast(fx[1] & ((@as(i64, 1) << @intCast(f.w)) - 1))) << @intCast(f.at);
            drop = true;
        };
        if (!drop) try fields.append(a, f);
    }
    if (m.bti) {
        try fields.append(a, .{ .n = "op2", .at = 5, .w = 3, .src = 0 });
        bits |= 4 << 8;
    }
    std.mem.sort(Field, fields.items, {}, struct {
        fn gt(_: void, x: Field, y: Field) bool {
            return x.at > y.at;
        }
    }.gt);
    return .{ .mn = m.mn, .bits = bits, .fields = fields.items, .ops = m.ops, .imm = m.imm, .args = m.args, .rec = try std.mem.concat(a, u8, &.{ m.rec, " (manual)" }) };
}

// ---- size merge ------------------------------------------------------------------------------

const arr_log = [_]struct { []const u8, i64 }{
    .{ "a8b", 0 }, .{ "a16b", 0 }, .{ "a4b", 0 }, .{ "a2b", 0 }, .{ "a4h", 1 }, .{ "a8h", 1 }, .{ "a2h", 1 },
    .{ "a2s", 2 }, .{ "a4s", 2 },  .{ "a1s", 2 }, .{ "a1d", 3 }, .{ "a2d", 3 }, .{ "a1q", 4 },
};
const et_log = [_]struct { []const u8, i64 }{ .{ "b", 0 }, .{ "h", 1 }, .{ "s", 2 }, .{ "d", 3 }, .{ "q", 4 } };
fn logOf(tab: anytype, s: []const u8) ?i64 {
    for (tab) |e| if (eql(u8, e[0], s)) return e[1];
    return null;
}

fn optEql(comptime T: type, x: ?T, y: ?T) bool {
    if (x == null or y == null) return x == null and y == null;
    if (T == []const u8) return eql(u8, x.?, y.?);
    return x.? == y.?;
}
fn opEql(x: Op, y: Op, strip: bool) bool {
    if (!eql(u8, x.k, y.k)) return false;
    const sa = strip and (eql(u8, x.k, "vfix") or eql(u8, x.k, "vlist"));
    const se = strip and (eql(u8, x.k, "zfix") or eql(u8, x.k, "pfix"));
    return optEql([]const u8, x.f, y.f) and optEql([]const u8, x.f2, y.f2) and (sa or optEql([]const u8, x.arr, y.arr)) and
        (se or optEql([]const u8, x.et, y.et)) and optEql([]const u8, x.sh, y.sh) and optEql([]const u8, x.mode, y.mode) and
        optEql([]const u8, x.mk, y.mk) and optEql(i64, x.n, y.n) and optEql(bool, x.w, y.w) and optEql(i64, x.fixed, y.fixed) and
        optEql(i64, x.scale, y.scale) and optEql(bool, x.signed, y.signed) and optEql(bool, x.opt, y.opt);
}
fn isListKind(k: []const u8) bool {
    return inList(&.{ "vfix", "vlist", "zfix", "pfix" }, k);
}
fn sameShape(x: Form, y: Form) bool {
    if (x.fields.len != y.fields.len or !eql(u8, x.imm, y.imm) or !std.mem.eql(i64, x.args, y.args) or x.ops.len != y.ops.len) return false;
    for (x.fields, y.fields) |p, q| if (!eql(u8, p.n, q.n) or p.at != q.at or p.w != q.w or p.src != q.src) return false;
    for (x.ops, y.ops) |p, q| if (!opEql(p, q, true)) return false;
    return true;
}
fn listEql(x: []const []const u8, y: []const []const u8) bool {
    if (x.len != y.len) return false;
    for (x, y) |p, q| if (!eql(u8, p, q)) return false;
    return true;
}

/// Merge records that differ only in the element-size field (bits 22-23) into one form with
/// element lists, indexed by the size field (a 1-bit SVE size field is stored as index - 2).
fn merge(a: Allocator, forms: []const Form) ![]Form {
    var out: std.ArrayList(Form) = .empty;
    const used = try a.alloc(bool, forms.len);
    @memset(used, false);
    for (forms, 0..) |fa, i| {
        if (used[i]) continue;
        var grp: std.ArrayList(usize) = .empty;
        try grp.append(a, i);
        for (forms[i + 1 ..], i + 1..) |fb, j| {
            if (used[j] or !sameShape(fa, fb)) continue;
            if ((fa.bits ^ fb.bits) & ~@as(u32, 3 << 22) != 0) continue;
            try grp.append(a, j);
        }
        const keep = struct {
            fn f(al: Allocator, o: *std.ArrayList(Form), fs: []const Form, g: []const usize, u: []bool) !void {
                for (g) |x| {
                    try o.append(al, fs[x]);
                    u[x] = true;
                }
            }
        }.f;
        if (grp.items.len == 1) {
            try out.append(a, fa);
            used[i] = true;
            continue;
        }
        var sve = false;
        var neon = false;
        for (fa.ops) |o| {
            if (eql(u8, o.k, "zfix") or eql(u8, o.k, "pfix")) sve = true;
            if (eql(u8, o.k, "vfix") or eql(u8, o.k, "vlist")) neon = true;
        }
        if (sve == neon) {
            try keep(a, &out, forms, grp.items, used);
            continue;
        }
        var diffall: u32 = 0;
        for (grp.items) |g| diffall |= forms[g].bits ^ fa.bits;
        const w: u5 = if (diffall & (1 << 23) != 0) 2 else 1;
        if (w == 1 and diffall & (1 << 22) == 0) {
            try keep(a, &out, forms, grp.items, used);
            continue;
        }
        const n: usize = @as(usize, 1) << w;
        // lists[op][size] = element
        const nops = fa.ops.len;
        const lists = try a.alloc([4]?[]const u8, nops);
        for (lists) |*l| l.* = .{ null, null, null, null };
        var ok = true;
        for (grp.items) |g| {
            const fm = forms[g];
            const sz: usize = (fm.bits >> 22) & ((@as(u32, 1) << w) - 1);
            if (!sve) {
                var mins: i64 = 99;
                for (fm.ops) |o| if (isListKind(o.k)) {
                    mins = @min(mins, logOf(arr_log, o.arr orelse o.et orelse "") orelse {
                        ok = false;
                        continue;
                    });
                };
                const expect = if (w == 2) mins else mins - 2;
                if (expect != @as(i64, @intCast(sz))) ok = false;
            }
            for (fm.ops, 0..) |o, pi| if (isListKind(o.k)) {
                if (lists[pi][sz] != null) ok = false;
                lists[pi][sz] = o.arr orelse o.et;
            };
        }
        if (!ok) {
            try keep(a, &out, forms, grp.items, used);
            continue;
        }
        // an operand whose element is the same in every member stays fixed (e.g. wide Zm.D)
        const is_list = try a.alloc(bool, nops);
        var nlists: usize = 0;
        for (fa.ops, 0..) |o, pi| {
            is_list[pi] = false;
            if (!isListKind(o.k)) continue;
            var first: ?[]const u8 = null;
            var varies = false;
            for (lists[pi]) |e| if (e) |x| {
                if (first) |fv| {
                    if (!eql(u8, fv, x)) varies = true;
                } else first = x;
            };
            if (varies) {
                is_list[pi] = true;
                nlists += 1;
            }
        }
        if (nlists == 0) {
            try keep(a, &out, forms, grp.items, used);
            continue;
        }
        const pad: usize = if (sve and w == 1) 2 else 0;
        const L = struct {
            fn f(al: Allocator, l: [4]?[]const u8, nn: usize, pd: usize) ![]const []const u8 {
                var r: std.ArrayList([]const u8) = .empty;
                for (0..pd) |_| try r.append(al, "none");
                for (0..nn) |k| try r.append(al, l[k] orelse "none");
                while (r.items.len > 1 and eql(u8, r.items[r.items.len - 1], "none")) r.items.len -= 1;
                return r.items;
            }
        }.f;
        var distinct: std.ArrayList([]const []const u8) = .empty;
        var first_list: ?[]const []const u8 = null;
        for (0..nops) |pi| if (is_list[pi]) {
            const l = try L(a, lists[pi], n, pad);
            if (first_list == null) first_list = l;
            var seen = false;
            for (distinct.items) |d| if (listEql(d, l)) {
                seen = true;
            };
            if (!seen) try distinct.append(a, l);
        };
        // the first list operand picks the size index, so its list must not repeat an element
        var dup = false;
        for (first_list.?, 0..) |x, xi| if (!eql(u8, x, "none")) for (first_list.?[xi + 1 ..]) |y| if (eql(u8, x, y)) {
            dup = true;
        };
        if (dup or distinct.items.len > 2) {
            try keep(a, &out, forms, grp.items, used);
            continue;
        }
        var m = fa;
        m.bits = fa.bits & ~(((@as(u32, 1) << w) - 1) << 22);
        var mf: std.ArrayList(Field) = .empty;
        try mf.appendSlice(a, fa.fields);
        try mf.append(a, .{ .n = "sz", .at = 22, .w = w, .src = 0 });
        std.mem.sort(Field, mf.items, {}, struct {
            fn gt(_: void, x: Field, y: Field) bool {
                return x.at > y.at;
            }
        }.gt);
        m.fields = mf.items;
        // one list: t / zt; two: the wider list is ta / zta, the narrower tb / ztb
        var names = [2][]const u8{ "", "" };
        if (distinct.items.len == 1) {
            names[0] = if (sve) "zt" else "t";
        } else {
            const key = struct {
                fn f(l: []const []const u8) i64 {
                    var mx: i64 = -1;
                    for (l) |x| if (!eql(u8, x, "none")) {
                        mx = @max(mx, logOf(arr_log, x) orelse logOf(et_log, x) orelse -1);
                    };
                    return mx;
                }
            }.f;
            const wider_first = key(distinct.items[0]) >= key(distinct.items[1]);
            names[0] = if (sve) "zta" else "ta";
            names[1] = if (sve) "ztb" else "tb";
            if (!wider_first) std.mem.swap([]const u8, &names[0], &names[1]);
        }
        const ops = try a.dupe(Op, fa.ops);
        for (ops, 0..) |*o, pi| if (is_list[pi]) {
            const l = try L(a, lists[pi], n, pad);
            const di: usize = if (listEql(distinct.items[0], l)) 0 else 1;
            const nm = names[di];
            if (eql(u8, o.k, "vfix")) {
                o.k = if (eql(u8, nm, "t")) "vt" else if (eql(u8, nm, "ta")) "vta" else "vtb";
                o.arr = null;
            } else if (eql(u8, o.k, "vlist")) {
                o.arr = null;
            } else if (eql(u8, o.k, "zfix")) {
                o.k = nm;
                o.et = null;
            } else if (eql(u8, o.k, "pfix")) {
                o.k = "pt";
                o.et = null;
            }
            inline for (.{ "t", "ta", "tb", "zt", "zta", "ztb" }) |fld| if (eql(u8, nm, fld)) {
                @field(m.lists, fld) = l;
            };
        };
        m.ops = ops;
        try out.append(a, m);
        for (grp.items) |g| used[g] = true;
    }
    return out.items;
}

// ---- output ----------------------------------------------------------------------------------

fn writeOp(a: Allocator, w: *std.ArrayList(u8), o: Op) !void {
    var first = true;
    try w.appendSlice(a, ".{ ");
    const P = struct {
        fn sep(al: Allocator, ww: *std.ArrayList(u8), f: *bool) !void {
            if (!f.*) try ww.appendSlice(al, ", ");
            f.* = false;
        }
    };
    inline for (.{ "k", "f", "f2", "arr", "et", "sh", "mode", "mk", "n", "w", "fixed", "scale", "signed", "opt" }) |key| {
        const v = @field(o, key);
        const T = @TypeOf(v);
        if (T == []const u8) {
            try P.sep(a, w, &first);
            try w.print(a, ".{s} = .{s}", .{ key, v });
        } else if (v) |x| {
            try P.sep(a, w, &first);
            const X = @TypeOf(x);
            if (X == []const u8) {
                if (comptime eql(u8, key, "f") or eql(u8, key, "f2")) try w.print(a, ".{s} = .@\"{s}\"", .{ key, x }) else try w.print(a, ".{s} = .{s}", .{ key, x });
            } else if (X == bool) {
                try w.print(a, ".{s} = {s}", .{ key, if (x) "true" else "false" });
            } else try w.print(a, ".{s} = {d}", .{ key, x });
        }
    }
    try w.appendSlice(a, " }");
}

fn formLine(a: Allocator, f: Form) ![]const u8 {
    var w: std.ArrayList(u8) = .empty;
    try w.print(a, "        .{{ .bits = 0x{X:0>8}, .fields = &.{{", .{f.bits});
    for (f.fields, 0..) |fl, i| {
        if (i > 0) try w.appendSlice(a, ", ");
        try w.print(a, ".{{ .n = .@\"{s}\", .at = {d}, .w = {d}, .src = {d} }}", .{ fl.n, fl.at, fl.w, fl.src });
    }
    try w.appendSlice(a, "}, .ops = &.{");
    for (f.ops, 0..) |o, i| {
        if (i > 0) try w.appendSlice(a, ", ");
        try writeOp(a, &w, o);
    }
    try w.appendSlice(a, "}");
    inline for (.{ "t", "ta", "tb", "zt", "zta", "ztb" }) |k| {
        const l = @field(f.lists, k);
        if (l.len > 0) {
            try w.print(a, ", .{s} = &.{{", .{k});
            for (l, 0..) |x, i| try w.print(a, "{s}.{s}", .{ if (i > 0) ", " else "", x });
            try w.appendSlice(a, "}");
        }
    }
    if (!eql(u8, f.imm, "none")) {
        try w.print(a, ", .imm = .{s}, .args = &.{{ ", .{f.imm});
        for (f.args, 0..) |x, i| try w.print(a, "{s}{d}", .{ if (i > 0) ", " else "", x });
        try w.appendSlice(a, " }");
    }
    if (f.inv_cond) try w.appendSlice(a, ", .inv_cond = true");
    try w.appendSlice(a, " },");
    return w.items;
}

fn dedupe(a: Allocator, forms: []const Form) ![]Form {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var out: std.ArrayList(Form) = .empty;
    for (forms) |f| {
        const gop = try seen.getOrPut(a, try formLine(a, f));
        if (gop.found_existing) continue;
        try out.append(a, f);
    }
    return out.items;
}

fn lessStr(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

const Table = std.StringArrayHashMapUnmanaged(std.ArrayList(Form));

fn add(a: Allocator, t: *Table, f: Form) !void {
    const gop = try t.getOrPut(a, f.mn);
    if (!gop.found_existing) gop.value_ptr.* = .empty;
    try gop.value_ptr.append(a, f);
}

/// All forms, per mnemonic, in table order.
pub fn generate(a: Allocator, db: lj.Db) !Table {
    var t: Table = .empty;
    for (db.instancesOf("Instruction")) |nv| {
        const n = nv.string;
        const r = db.get(n) orelse continue;
        if (lj.int(r, "isPseudo") != 0 or lj.int(r, "isCodeGenOnly") != 0 or lj.int(r, "Size") != 4) continue;
        if (lj.str(r, "AsmString").len == 0) continue;
        if (skippedRecord(n)) continue;
        const rec = (try recordOf(a, r)) orelse continue;
        const fs = build(a, n, &rec) catch |e| switch (e) {
            error.Skip => continue,
            else => return e,
        };
        for (fs) |f| try add(a, &t, f);
    }
    for (db.instancesOf("InstAlias")) |nv| {
        const n = nv.string;
        const al = db.get(n) orelse continue;
        if (lj.str(al, "AsmVariantName").len != 0) continue;
        var tgt: []const u8 = "";
        const rec = aliasRecord(a, db, al, &tgt) catch |e| switch (e) {
            error.Skip => continue,
            else => return e,
        };
        if (skippedRecord(tgt)) continue;
        const name = try std.fmt.allocPrint(a, "{s} (alias {s})", .{ tgt, n });
        const fs = build(a, name, &rec) catch |e| switch (e) {
            error.Skip => continue,
            else => return e,
        };
        for (fs) |f| try add(a, &t, f);
    }
    // manual forms first, in list order
    var i = manual.len;
    while (i > 0) {
        i -= 1;
        const f = try manualForm(a, db, manual[i]);
        const gop = try t.getOrPut(a, f.mn);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.insert(a, 0, f);
    }
    for (src_w) |mn| if (t.getPtr(mn)) |l| for (l.items) |*f| {
        if (f.ops.len == 2 and eql(u8, f.ops[0].k, "x") and eql(u8, f.ops[1].k, "x")) {
            const ops = try a.dupe(Op, f.ops);
            ops[1].k = "w";
            f.ops = ops;
        }
    };
    for (t.values()) |*l| {
        const fs = try dedupe(a, try merge(a, try dedupe(a, l.items)));
        var hi: std.ArrayList(Form) = .empty;
        var lo: std.ArrayList(Form) = .empty;
        for (fs) |f| try (if (lowPriority(f)) &lo else &hi).append(a, f);
        try hi.appendSlice(a, lo.items);
        l.* = hi;
    }
    for (kept_mnemonics) |m| {
        const gop = try t.getOrPut(a, m);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
    }
    return t;
}

pub fn render(a: Allocator, db: lj.Db, header: []const u8) ![]const u8 {
    const t = try generate(a, db);
    var names: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (extra_names) |n| try names.put(a, n, {});
    var fns: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (imm_fns) |n| try fns.put(a, n, {});
    for (t.values()) |l| for (l.items) |f| {
        for (f.fields) |fl| try names.put(a, fl.n, {});
        for (f.ops) |o| {
            if (o.f) |x| try names.put(a, x, {});
            if (o.f2) |x| try names.put(a, x, {});
        }
        if (!fns.contains(f.imm)) {
            std.debug.print("gen_a64: ImmFn {s} ({s}) is not implemented in a64.zig\n", .{ f.imm, f.rec });
            return error.UnknownImmFn;
        }
    };
    const sortKeys = struct {
        fn f(al: Allocator, keys: []const []const u8) ![]const []const u8 {
            const k = try al.dupe([]const u8, keys);
            std.mem.sort([]const u8, k, {}, lessStr);
            return k;
        }
    }.f;
    var w: std.ArrayList(u8) = .empty;
    try w.appendSlice(a, header);
    try w.appendSlice(a, "const a64 = @import(\"a64.zig\");\nconst Form = a64.Form;\n\npub const Name = enum(u8) {\n");
    for (try sortKeys(a, names.keys())) |n| try w.print(a, "    @\"{s}\",\n", .{n});
    try w.appendSlice(a, "};\n\npub const ImmFn = enum {\n");
    for (try sortKeys(a, fns.keys())) |n| try w.print(a, "    {s},\n", .{n});
    try w.appendSlice(a, "};\n\npub const Mnemonic = enum {\n");
    const mns = try sortKeys(a, t.keys());
    for (mns) |m| try w.print(a, "    @\"{s}\",\n", .{m});
    try w.appendSlice(a, "};\n\npub const table = struct {\n");
    for (mns) |m| {
        try w.print(a, "    pub const @\"{s}\" = [_]Form{{\n", .{m});
        for (t.get(m).?.items) |f| {
            try w.print(a, "        // {s}\n", .{f.rec});
            try w.appendSlice(a, try formLine(a, f));
            try w.append(a, '\n');
        }
        try w.appendSlice(a, "    };\n");
    }
    try w.appendSlice(a, "};\n");
    return w.items;
}
