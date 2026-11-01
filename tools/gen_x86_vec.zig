//! x86-64 non-GP forms (src/x86_64/vec_table.zig: SSE, AVX, AVX-512, AVX10, MMX/3DNow!, x87, AMX,
//! mask, crypto and the system/segment forms) from LLVM's X86 target description
//! (`llvm-tblgen --dump-json X86.td`).
//!
//! Every LLVM instruction record with a VEX, XOP or EVEX encoding, and every legacy record whose
//! mnemonic is not in gp_table.zig, becomes a form: operand kinds in Intel order, prefixes, map,
//! opcode, VEX/EVEX pp/W/L, ModRM roles (reg, rm, vvvv, is4, opcode register), immediate size,
//! disp8*N and the CPU mode. LLVM records an EVEX instruction once per masking / broadcast /
//! rounding variant (`rr`, `rrk`, `rrkz`, `rmb`, `rrb`, ...); these are merged into one form with
//! the opmask (`{k}`/`{z}`), embedded-broadcast element size and `{er}`/`{sae}` flags that vec.zig
//! expects. Condition-code records (`cmp${cond}xadd`) are expanded into the 16 mnemonics.
//!
//! Project decisions that are not LLVM data are the small commented lists at the end of this file:
//! mnemonic spellings, mnemonics not offered, the GP-table mnemonics that also have forms here,
//! a few forms LLVM does not define, and the form-selection hints (`lowprio`, `late_vex`).
const std = @import("std");
const lj = @import("llvm_json.zig");
const gp = @import("gen_x86_gp.zig");

const Enc = enum { legacy, vex, xop, evex };

pub const Op = struct { k: []const u8, size: u16 = 0, bcst: u8 = 0 };

pub const Form = struct {
    mn: []const u8,
    ops: []const Op,
    enc: Enc = .legacy,
    pre: []const u8 = &.{},
    rexw: bool = false,
    map: u8 = 0,
    op: []const u8,
    digit: ?u8 = null,
    imm: u16 = 0,
    pp: u8 = 0,
    w: u8 = 2,
    l: u8 = 0,
    reg: ?usize = null,
    rm: ?usize = null,
    vvvv: ?usize = null,
    is4: ?usize = null,
    plus: ?usize = null,
    mask: u8 = 0,
    er: bool = false,
    sae: bool = false,
    n: u64 = 1,
    /// broadcast count of an `rmb` record (0: none); only used while merging
    bc: u64 = 0,
    lowprio: bool = false,
    late_vex: bool = false,
    exact_mem: bool = false,
    mode: []const u8 = "any",
};

const Skip = error{Skip};
const eql = std.mem.eql;

fn has(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (eql(u8, x, s)) return true;
    return false;
}

/// Intel variant (`{att|intel}`) of an LLVM asm string. Literal braces inside a variant
/// (`{${mask}}`, `{z}`, `{sae}`, `{1to4}`) are kept; a one-alternative group outside a variant is
/// an AT&T-only suffix and is dropped.
fn intel(a: std.mem.Allocator, s: []const u8, inner: bool) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c == '$' and i + 1 < s.len and s[i + 1] == '{') {
            const e = std.mem.indexOfScalarPos(u8, s, i, '}').?;
            try out.appendSlice(a, s[i .. e + 1]);
            i = e + 1;
        } else if (c == '\\' and i + 1 < s.len) {
            try out.append(a, s[i + 1]);
            i += 2;
        } else if (c == '{') {
            var depth: usize = 0;
            var j = i + 1;
            var alts: std.ArrayList([]const u8) = .empty;
            var st = j;
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
                    try alts.append(a, s[st..j]);
                    st = j + 1;
                }
            }
            try alts.append(a, s[st..@min(j, s.len)]);
            if (alts.items.len > 1) {
                try out.appendSlice(a, try intel(a, alts.items[1], true));
            } else if (inner and literalBrace(alts.items[0])) {
                try out.print(a, "{{{s}}}", .{alts.items[0]});
            }
            i = j + 1;
        } else {
            try out.append(a, c);
            i += 1;
        }
    }
    return out.items;
}

fn literalBrace(x: []const u8) bool {
    if (eql(u8, x, "z") or eql(u8, x, "sae")) return true;
    if (std.mem.startsWith(u8, x, "1to")) {
        for (x[3..]) |c| if (!std.ascii.isDigit(c)) return false;
        return x.len > 3;
    }
    if (x.len > 1 and x[0] == '$') {
        var y = x[1..];
        if (y.len > 1 and y[0] == '{' and y[y.len - 1] == '}') y = y[1 .. y.len - 1];
        for (y) |c| if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
        return y.len > 0;
    }
    return false;
}

// ---- operand classes -------------------------------------------------------------------------

const reg_classes = [_][2][]const u8{
    .{ "VR128", "xmm" },      .{ "VR128X", "xmm" },       .{ "FR32", "xmm" },         .{ "FR64", "xmm" },
    .{ "FR16X", "xmm" },      .{ "FR32X", "xmm" },        .{ "FR64X", "xmm" },        .{ "VR256", "ymm" },
    .{ "VR256X", "ymm" },     .{ "VR512", "zmm" },        .{ "VR64", "mm" },          .{ "TILE", "tmm" },
    .{ "RSTi", "sti" },       .{ "GR8", "r8" },           .{ "GR16", "r16" },         .{ "GR32", "r32" },
    .{ "GR64", "r64" },       .{ "SEGMENT_REG", "sreg" }, .{ "CONTROL_REG", "creg" }, .{ "DEBUG_REG", "dreg" },
    .{ "GR32orGR64", "r32" },
};
const mem_classes = [_]struct { []const u8, u16 }{
    .{ "f128mem", 16 },      .{ "i128mem", 16 },      .{ "f256mem", 32 },      .{ "i256mem", 32 },  .{ "f512mem", 64 },
    .{ "i512mem", 64 },      .{ "f16mem", 2 },        .{ "i16mem", 2 },        .{ "shmem", 2 },     .{ "f32mem", 4 },
    .{ "i32mem", 4 },        .{ "ssmem", 4 },         .{ "f64mem", 8 },        .{ "i64mem", 8 },    .{ "sdmem", 8 },
    .{ "f80mem", 10 },       .{ "i8mem", 1 },         .{ "anymem", 0 },        .{ "opaquemem", 0 }, .{ "sibmem", 0 },
    .{ "i512mem_GR16", 64 }, .{ "i512mem_GR32", 64 }, .{ "i512mem_GR64", 64 },
};
const imm_classes = [_]struct { []const u8, u16 }{
    .{ "u8imm", 1 },    .{ "i8imm", 1 },     .{ "i32u8imm", 1 }, .{ "i16u8imm", 1 }, .{ "i64u8imm", 1 },
    .{ "i16i8imm", 1 }, .{ "i32i8imm", 1 },  .{ "i64i8imm", 1 }, .{ "u4imm", 1 },    .{ "i16imm", 2 },
    .{ "i32imm", 4 },   .{ "i64i32imm", 4 }, .{ "i64imm", 8 },
};

fn isImmClass(cls: []const u8) bool {
    for (imm_classes) |x| if (eql(u8, x[0], cls)) return true;
    return false;
}

fn kindOf(cls: []const u8) !Op {
    for (reg_classes) |x| if (eql(u8, x[0], cls)) return .{ .k = x[1] };
    if (std.mem.startsWith(u8, cls, "VK")) return .{ .k = "k" };
    for (mem_classes) |x| if (eql(u8, x[0], cls)) return .{ .k = "mem", .size = x[1] };
    // VSIB: v{x,y,z}{32,64}[x]mem (the index size is fixed up from the mnemonic)
    if (cls.len >= 7 and cls[0] == 'v' and std.mem.endsWith(u8, cls, "mem") and (cls[1] == 'x' or cls[1] == 'y' or cls[1] == 'z')) {
        const k: []const u8 = switch (cls[1]) {
            'x' => "vx",
            'y' => "vy",
            else => "vz",
        };
        return .{ .k = k, .size = if (std.mem.startsWith(u8, cls[2..], "32")) 4 else 8 };
    }
    for (imm_classes) |x| if (eql(u8, x[0], cls)) return .{ .k = "imm", .size = x[1] };
    // string instructions: srcidx8 / dstidx64 ...
    for ([_][]const u8{ "srcidx", "dstidx" }) |p| if (std.mem.startsWith(u8, cls, p)) {
        const bits_ = std.fmt.parseInt(u16, cls[p.len..], 10) catch return error.UnknownClass;
        return .{ .k = "mem", .size = bits_ / 8 };
    };
    // GP-only operands (branch targets, moffs, lea addresses)
    if (std.mem.indexOf(u8, cls, "brtarget") != null or std.mem.startsWith(u8, cls, "offset") or
        std.mem.startsWith(u8, cls, "lea") or eql(u8, cls, "GR16orGR32orGR64")) return error.Skip;
    std.debug.print("gen_x86_vec: unknown operand class {s}\n", .{cls});
    return error.UnknownClass;
}

/// Fixed registers spelled literally in Intel asm strings.
const literal_regs = [_][2][]const u8{
    .{ "ds", "ds" },   .{ "es", "es" },     .{ "fs", "fs" }, .{ "gs", "gs" }, .{ "ss", "ss" }, .{ "cs", "cs" },
    .{ "st", "st0" },  .{ "st(0)", "st0" }, .{ "ax", "ax" }, .{ "al", "al" }, .{ "cl", "cl" }, .{ "eax", "eax" },
    .{ "rax", "rax" }, .{ "dx", "dx" },
};

fn mapOf(m: []const u8) !i16 {
    const t = [_]struct { []const u8, i16 }{
        .{ "OB", 0 },   .{ "TB", 1 },    .{ "T8", 2 },     .{ "TA", 3 },     .{ "ThreeDNow", 4 }, .{ "XOP8", 8 },
        .{ "XOP9", 9 }, .{ "XOPA", 10 }, .{ "T_MAP5", 5 }, .{ "T_MAP6", 6 }, .{ "T_MAP7", 7 },    .{ "T_MAP4", -1 },
    };
    for (t) |x| if (eql(u8, x[0], m)) return x[1];
    return error.UnknownMap;
}

fn ppOf(p: []const u8) u8 {
    if (eql(u8, p, "PD")) return 1;
    if (eql(u8, p, "XS")) return 2;
    if (eql(u8, p, "XD")) return 3;
    return 0;
}

fn predicates(a: std.mem.Allocator, r: lj.Obj) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (lj.list(r, "Predicates")) |p| try out.append(a, lj.defName(p));
    return out.items;
}

// ---- one record ------------------------------------------------------------------------------

const Tied = struct {
    pairs: [][2][]const u8,
    fn partner(t: Tied, n: []const u8) ?[]const u8 {
        for (t.pairs) |p| {
            if (eql(u8, p[0], n)) return p[1];
            if (eql(u8, p[1], n)) return p[0];
        }
        return null;
    }
};

fn roleSeq(fm: []const u8, v4: bool, imm8reg: bool, out: *[4][]const u8) []const []const u8 {
    var n: usize = 0;
    const push = struct {
        fn f(o: *[4][]const u8, i: *usize, s: []const u8) void {
            o[i.*] = s;
            i.* += 1;
        }
    }.f;
    if (std.mem.startsWith(u8, fm, "MRMDest") and std.mem.indexOf(u8, fm, "4VOp3") == null) {
        push(out, &n, "rm");
        if (v4) push(out, &n, "vvvv");
        push(out, &n, "reg");
    } else if (eql(u8, fm, "MRMSrcReg4VOp3") or eql(u8, fm, "MRMSrcMem4VOp3") or std.mem.startsWith(u8, fm, "MRMDestMem4VOp3")) {
        push(out, &n, "reg");
        push(out, &n, "rm");
        push(out, &n, "vvvv");
    } else if (eql(u8, fm, "MRMSrcRegOp4") or eql(u8, fm, "MRMSrcMemOp4")) {
        push(out, &n, "reg");
        push(out, &n, "vvvv");
        push(out, &n, "is4");
        push(out, &n, "rm");
    } else if (std.mem.startsWith(u8, fm, "MRMSrc")) {
        push(out, &n, "reg");
        if (v4) push(out, &n, "vvvv");
        push(out, &n, "rm");
        if (imm8reg) push(out, &n, "is4");
    } else if (fm.len == 5 and std.mem.startsWith(u8, fm, "MRM") and ((fm[3] >= '0' and fm[3] <= '7') or fm[3] == 'X') and (fm[4] == 'r' or fm[4] == 'm')) {
        if (v4) push(out, &n, "vvvv");
        push(out, &n, "rm");
    } else if (eql(u8, fm, "MRMr0")) { // tilezero: ModRM.reg, mod=11 rm=0
        push(out, &n, "reg");
    } else if (eql(u8, fm, "AddRegFrm")) {
        push(out, &n, "plus");
    }
    return out[0..n];
}

fn fromRecord(a: std.mem.Allocator, rname: []const u8, r: lj.Obj) !Form {
    const s = std.mem.trim(u8, try intel(a, lj.str(r, "AsmString"), false), " \t");
    const sp = std.mem.indexOfAny(u8, s, " \t") orelse s.len;
    // `{l}call{l}`: the Intel spelling of the m16:32 far call/jump is plain call/jmp
    const mn = if (eql(u8, rname, "FARCALL32m")) "lcall" else if (eql(u8, rname, "FARJMP32m")) "ljmp" else try std.ascii.allocLowerString(a, s[0..sp]);
    const rest = std.mem.trim(u8, s[sp..], " \t");
    const preds = try predicates(a, r);

    const mp = try mapOf(lj.def(r, "OpMap"));
    if (mp < 0) return error.Skip; // EVEX map 4: APX
    const ad = lj.def(r, "AdSize");
    if (eql(u8, ad, "AdSize16") or (eql(u8, ad, "AdSize32") and has(preds, "In64BitMode"))) return error.Skip;
    if (std.mem.startsWith(u8, rname, "FARCALL") or std.mem.startsWith(u8, rname, "FARJMP")) {
        if (mn.len == 0 or mn[0] != 'l') return error.Skip;
    }
    // APX promotions (`_NF`, `_EVEX`, `_ND`) and extended-GPR forms
    for ([_][]const u8{ "_NF", "_EVEX", "_ND", "_NF_ND" }) |suf| if (std.mem.endsWith(u8, rname, suf)) return error.Skip;
    if (lj.int(r, "hasEVEX_NF") != 0 or has(preds, "HasEGPR") or has(preds, "HasNDD")) return error.Skip;

    const outs = try lj.dagArgs(a, r, "OutOperandList");
    const ins = try lj.dagArgs(a, r, "InOperandList");
    var all: std.ArrayList(lj.DagArg) = .empty;
    try all.appendSlice(a, outs);
    try all.appendSlice(a, ins);
    const classOf = struct {
        fn f(xs: []const lj.DagArg, n: []const u8) ?[]const u8 {
            var c: ?[]const u8 = null;
            for (xs) |x| if (eql(u8, x.name, n)) {
                c = x.class;
            };
            return c;
        }
    }.f;
    const enc: Enc = blk: {
        const e = lj.def(r, "OpEnc");
        break :blk if (eql(u8, e, "EncVEX")) .vex else if (eql(u8, e, "EncXOP")) .xop else if (eql(u8, e, "EncEVEX")) .evex else .legacy;
    };

    // asm operands (Intel order)
    var ops: std.ArrayList(Op) = .empty;
    var opnames: std.ArrayList(?[]const u8) = .empty;
    var mask: u8 = 0;
    var er = false;
    var sae = false;
    var bc: u64 = 0;
    if (rest.len != 0) {
        var it = std.mem.splitScalar(u8, rest, ',');
        while (it.next()) |t0| {
            const t = std.mem.trim(u8, t0, " \t");
            if (std.mem.indexOf(u8, t, "{z}") != null) mask = 2;
            if (std.mem.indexOf(u8, t, "{${mask}}") != null or std.mem.indexOf(u8, t, "{$mask}") != null) mask = @max(mask, 1);
            if (std.mem.indexOf(u8, t, "{sae}") != null) sae = true;
            var bcount: u64 = 0;
            if (std.mem.indexOf(u8, t, "{1to")) |at| {
                const e = std.mem.indexOfScalarPos(u8, t, at, '}').?;
                bcount = try std.fmt.parseInt(u64, t[at + 4 .. e], 10);
            }
            // strip `{${mask}}`, `{z}`, `{sae}`, `{1toN}`
            var core: std.ArrayList(u8) = .empty;
            var i: usize = 0;
            while (i < t.len) {
                if (t[i] == '$' and i + 1 < t.len and t[i + 1] == '{') {
                    const e = std.mem.indexOfScalarPos(u8, t, i, '}').?;
                    try core.appendSlice(a, t[i .. e + 1]);
                    i = e + 1;
                } else if (t[i] == '{') {
                    var e = std.mem.indexOfScalarPos(u8, t, i, '}').?;
                    if (i + 2 < t.len and t[i + 1] == '$' and t[i + 2] == '{') e = std.mem.indexOfScalarPos(u8, t, e + 1, '}').?;
                    i = e + 1;
                } else {
                    try core.append(a, t[i]);
                    i += 1;
                }
            }
            const c = std.mem.trim(u8, core.items, " \t");
            if (c.len == 0) continue;
            if (c[0] != '$') {
                const lc = try std.ascii.allocLowerString(a, c);
                if (eql(u8, lc, "xmm0")) continue; // implicit operand
                var found: ?[]const u8 = null;
                for (literal_regs) |x| if (eql(u8, x[0], lc)) {
                    found = x[1];
                };
                const k = found orelse {
                    std.debug.print("gen_x86_vec: {s}: literal operand {s}\n", .{ rname, c });
                    return error.UnknownOperand;
                };
                try ops.append(a, .{ .k = k });
                try opnames.append(a, null);
                continue;
            }
            var nm = c[1..];
            if (nm.len > 1 and nm[0] == '{') nm = nm[1 .. nm.len - 1];
            const cls = classOf(all.items, nm) orelse return error.UnknownOperand;
            if (eql(u8, cls, "AVX512RC")) {
                er = true;
                continue;
            }
            var k = try kindOf(cls);
            if (bcount != 0) {
                k.bcst = @intCast(k.size);
                k.size = 0;
                bc = bcount;
            }
            try ops.append(a, k);
            try opnames.append(a, nm);
        }
    }

    const fm = lj.def(r, "Form");
    const v4 = lj.int(r, "hasVEX_4V") != 0;
    const imm8reg = eql(u8, lj.def(r, "ImmT"), "Imm8Reg");
    const tied = Tied{ .pairs = try lj.tiedPairs(a, r) };
    // encoding operand order: MI order, a tied pair at its last position, mask/rc/imm dropped
    var order: std.ArrayList([]const u8) = .empty;
    for (all.items, 0..) |x, i| {
        if (eql(u8, x.class, "AVX512RC") or isImmClass(x.class)) continue;
        if (enc == .evex and std.mem.startsWith(u8, x.class, "VK") and std.mem.endsWith(u8, x.class, "WM") and eql(u8, x.name, "mask")) continue;
        if (tied.partner(x.name)) |p| {
            var later = false;
            for (all.items[i + 1 ..]) |y| if (eql(u8, y.name, p)) {
                later = true;
            };
            if (later) continue;
        }
        try order.append(a, x.name);
    }
    var seq_buf: [4][]const u8 = undefined;
    const seq = roleSeq(fm, v4, imm8reg, &seq_buf);
    const Roles = struct {
        names: []const []const u8,
        seq: []const []const u8,
        opnames: []const ?[]const u8,
        tied: Tied,
        fn idx(self: @This(), ro: []const u8) ?usize {
            var nm: ?[]const u8 = null;
            for (self.seq, 0..) |x, i| if (eql(u8, x, ro) and i < self.names.len) {
                nm = self.names[i];
            };
            const n = nm orelse return null;
            for (self.opnames, 0..) |o, i| if (o) |on| {
                if (eql(u8, on, n)) return i;
                if (self.tied.partner(on)) |p| if (eql(u8, p, n)) return i;
            };
            return null;
        }
    };
    const roles = Roles{ .names = order.items, .seq = seq, .opnames = opnames.items, .tied = tied };

    var op: std.ArrayList(u8) = .empty;
    try op.append(a, @intCast(lj.bits(r, "Opcode")));
    var digit: ?u8 = null;
    if (fm.len == 5 and std.mem.startsWith(u8, fm, "MRM") and fm[3] >= '0' and fm[3] <= '7' and (fm[4] == 'r' or fm[4] == 'm')) digit = fm[3] - '0';
    if (std.mem.startsWith(u8, fm, "MRM_")) {
        const xx = try std.fmt.parseInt(u8, fm[4..], 16);
        // fixed ModRM byte: `/digit` with mod=11, rm=0 when the form has an immediate or a
        // VEX/EVEX prefix (xabort, hreset, tilerelease), an extra opcode byte otherwise
        if (xx & 7 == 0 and (enc != .legacy or !eql(u8, lj.def(r, "ImmT"), "NoImm"))) digit = (xx >> 3) & 7 else try op.append(a, xx);
    }
    var pre: std.ArrayList(u8) = .empty;
    var pp = ppOf(lj.def(r, "OpPrefix"));
    var rexw = false;
    var w: u8 = 2;
    if (enc == .legacy) {
        if (eql(u8, lj.def(r, "OpSize"), "OpSize16")) try pre.append(a, 0x66);
        if (pp != 0) try pre.append(a, ([_]u8{ 0, 0x66, 0xF3, 0xF2 })[pp]);
        pp = 0;
        rexw = lj.int(r, "hasREX_W") != 0;
    } else {
        w = if (lj.int(r, "hasREX_W") != 0) 1 else if (lj.int(r, "IgnoresW") != 0) 2 else 0;
    }
    var l: u8 = if (lj.int(r, "hasEVEX_L2") != 0) 2 else if (lj.int(r, "hasVEX_L") != 0) 1 else 0;
    if (er or sae) { // EVEX.L'L holds the rounding mode: the vector length is that of the operands
        var z = false;
        var y = false;
        for (ops.items) |o| {
            if (eql(u8, o.k, "zmm")) z = true;
            if (eql(u8, o.k, "ymm")) y = true;
        }
        if (z) l = 2 else if (y) l = 1;
    }
    var plus = roles.idx("plus");
    var rm = roles.idx("rm");
    const reg = roles.idx("reg");
    var has_sti = false;
    for (ops.items) |o| if (eql(u8, o.k, "sti")) {
        has_sti = true;
    };
    if (has_sti and digit != null and std.mem.endsWith(u8, fm, "r")) {
        // x87 `st(i)` forms: opcode + (C0 | digit << 3) + i
        const b0 = op.items[0];
        op.clearRetainingCapacity();
        try op.appendSlice(a, &.{ b0, 0xC0 | (digit.? << 3) });
        plus = rm;
        rm = null;
        digit = null;
    }
    // VSIB: the operand size is the index element size (vgatherdpd: dword indices)
    if (vsibIndex(mn)) |isz| for (ops.items) |*o| {
        if (eql(u8, o.k, "vx") or eql(u8, o.k, "vy") or eql(u8, o.k, "vz")) o.size = isz;
    };
    // far pointer in memory: m16:16 / m16:32 / m16:64
    if (std.mem.startsWith(u8, fm, "MRM") and (eql(u8, mn, "lcall") or eql(u8, mn, "ljmp")) and ops.items.len > 0 and eql(u8, ops.items[0].k, "mem")) {
        const has66 = std.mem.indexOfScalar(u8, pre.items, 0x66) != null;
        ops.items[0] = .{ .k = "mem", .size = if (lj.int(r, "hasREX_W") != 0) 10 else if (has66) 4 else 6 };
        ops.shrinkRetainingCapacity(1);
    }
    // address held in a GP register (umonitor [reg], movdir64b/enqcmd es:[reg]): a memory operand
    // in this table, one form per mode
    var mode_override: ?[]const u8 = null;
    if (has(&addr_in_reg, mn) and ops.items.len > 0 and (eql(u8, ops.items[0].k, "r16") or eql(u8, ops.items[0].k, "r32") or eql(u8, ops.items[0].k, "r64"))) {
        if (eql(u8, ops.items[0].k, "r16")) return error.Skip;
        mode_override = if (eql(u8, ops.items[0].k, "r32")) "x86" else "x64";
        ops.items[0] = if (eql(u8, mn, "umonitor")) .{ .k = "maddr" } else .{ .k = "mem", .size = 64 };
    }
    // SDM: RSTORSSP m64, CLRSSBSY m64 (LLVM: i32mem)
    if ((eql(u8, mn, "rstorssp") or eql(u8, mn, "clrssbsy")) and ops.items.len == 1) ops.items[0].size = 8;
    // `call m16` (66 FF /2) would be picked for an unsized `call [eax]`
    if (eql(u8, mn, "call") and std.mem.indexOfScalar(u8, pre.items, 0x66) != null and ops.items.len == 1 and eql(u8, ops.items[0].k, "mem")) return error.Skip;
    // SDM: VMOVW m16 (LLVM's AVX10.2 VMOVZPWILo2PWIZmr says i32mem)
    if (eql(u8, mn, "vmovw")) for (ops.items) |*o| {
        if (eql(u8, o.k, "mem")) o.size = 2;
    };
    var imm: u16 = 0;
    if (!imm8reg) for (ops.items) |o| if (eql(u8, o.k, "imm")) {
        imm += o.size;
    };
    const n_raw: u64 = if (enc == .evex) lj.bits(r, "CD8_Scale") else 1;
    var f = Form{
        .mn = mn,
        .ops = ops.items,
        .enc = enc,
        .pre = pre.items,
        .rexw = rexw,
        .map = @intCast(mp),
        .op = op.items,
        .digit = digit,
        .imm = imm,
        .pp = pp,
        .w = w,
        .l = l,
        .reg = reg,
        .rm = rm,
        .vvvv = roles.idx("vvvv"),
        .is4 = roles.idx("is4"),
        .plus = plus,
        .mask = mask,
        .er = er,
        .sae = sae or er,
        .n = if (n_raw == 0) 1 else n_raw,
        .bc = bc,
    };
    // CMPccXADD is 64-bit only in the SDM and in LLVM; the table keeps it in every mode as before
    // (project decision: the AsmJit x86 tests encode it in 32-bit mode)
    if (has(preds, "In64BitMode") and !eql(u8, mn, "cmp${cond}xadd")) f.mode = "x64";
    if (has(preds, "Not64BitMode")) f.mode = "x86";
    if (mode_override) |m| f.mode = m;
    return f;
}

/// Index element size of a gather/scatter (`vgatherdpd`: d = dword indices).
fn vsibIndex(mn: []const u8) ?u16 {
    for ([_][]const u8{ "gather", "scatter" }) |p| if (std.mem.indexOf(u8, mn, p)) |at| {
        var i = at + p.len;
        if (i + 3 <= mn.len and eql(u8, mn[i .. i + 2], "pf") and (mn[i + 2] == '0' or mn[i + 2] == '1')) i += 3;
        if (i < mn.len and mn[i] == 'd') return 4;
        if (i < mn.len and mn[i] == 'q') return 8;
    };
    return null;
}

// ---- rendering -------------------------------------------------------------------------------

fn optStr(a: std.mem.Allocator, v: anytype) ![]const u8 {
    return if (v) |x| try std.fmt.allocPrint(a, "{d}", .{x}) else "null";
}

fn hexList(a: std.mem.Allocator, bs: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (bs, 0..) |b, i| {
        if (i != 0) try out.appendSlice(a, ", ");
        try out.print(a, "0x{x:0>2}", .{b});
    }
    return out.items;
}

pub fn line(a: std.mem.Allocator, f: Form) ![]const u8 {
    var ops: std.ArrayList(u8) = .empty;
    for (f.ops, 0..) |o, i| {
        if (i != 0) try ops.appendSlice(a, ", ");
        try ops.print(a, ".{{ .k = .{s}, .size = {d}, .bcst = {d} }}", .{ o.k, o.size, o.bcst });
    }
    const mode = if (eql(u8, f.mode, "any")) "" else try std.fmt.allocPrint(a, ", .mode = .{s}", .{f.mode});
    return std.fmt.allocPrint(a, ".{{ .ops = &.{{{s}}}, .enc = .{s}, .pre = &.{{{s}}}, .rexw = {}, .map = {d}, .op = &.{{{s}}}, .digit = {s}, .imm = {d}, .pp = {d}, .w = {d}, .l = {d}, .reg = {s}, .rm = {s}, .vvvv = {s}, .is4 = {s}, .plus = {s}, .mask = {d}, .er = {}, .sae = {}, .n = {d}, .lowprio = {}, .late_vex = {}, .exact_mem = {}{s} }}", .{
        ops.items,              @tagName(f.enc),       try hexList(a, f.pre), f.rexw,                f.map,       try hexList(a, f.op),
        try optStr(a, f.digit), f.imm,                 f.pp,                  f.w,                   f.l,         try optStr(a, f.reg),
        try optStr(a, f.rm),    try optStr(a, f.vvvv), try optStr(a, f.is4),  try optStr(a, f.plus), f.mask,      f.er,
        f.sae,                  f.n,                   f.lowprio,             f.late_vex,            f.exact_mem, mode,
    });
}

// ---- building the table ----------------------------------------------------------------------

const Item = struct { rec: []const u8, f: Form };

fn opsEql(x: []const Op, y: []const Op) bool {
    if (x.len != y.len) return false;
    for (x, y) |p, q| if (!eql(u8, p.k, q.k) or p.size != q.size or p.bcst != q.bcst) return false;
    return true;
}

fn optEql(x: ?usize, y: ?usize) bool {
    if (x == null or y == null) return x == null and y == null;
    return x.? == y.?;
}

fn optEql8(x: ?u8, y: ?u8) bool {
    if (x == null or y == null) return x == null and y == null;
    return x.? == y.?;
}

/// Same encoding apart from the operands: the register and memory forms of one instruction and
/// vector length.
fn sameEncoding(x: Form, y: Form) bool {
    return x.enc == y.enc and x.map == y.map and eql(u8, x.op, y.op) and x.pp == y.pp and x.w == y.w and
        x.l == y.l and eql(u8, x.pre, y.pre) and x.rexw == y.rexw and optEql8(x.digit, y.digit);
}

/// Merge key: everything but the operand sizes, opmask, broadcast and rounding.
fn sameGroup(x: Form, y: Form) bool {
    if (!sameEncoding(x, y) or x.imm != y.imm or !eql(u8, x.mode, y.mode)) return false;
    if (!optEql(x.reg, y.reg) or !optEql(x.rm, y.rm) or !optEql(x.vvvv, y.vvvv) or !optEql(x.is4, y.is4) or !optEql(x.plus, y.plus)) return false;
    if (x.ops.len != y.ops.len) return false;
    for (x.ops, y.ops) |p, q| if (!eql(u8, p.k, q.k)) return false;
    return true;
}

fn isMem(f: Form) bool {
    for (f.ops) |o| if (eql(u8, o.k, "mem") or eql(u8, o.k, "vx") or eql(u8, o.k, "vy") or eql(u8, o.k, "vz")) return true;
    return false;
}

fn expandCond(a: std.mem.Allocator, f: Form, out: *std.ArrayList(Form)) !void {
    const at = std.mem.indexOf(u8, f.mn, "${cond}") orelse return out.append(a, f);
    for (gp.cc_names, 0..) |cc, i| {
        var g = f;
        g.mn = try std.mem.concat(a, u8, &.{ f.mn[0..at], cc, f.mn[at + 7 ..] });
        const op = try a.dupe(u8, f.op);
        op[0] += @intCast(i);
        g.op = op;
        try out.append(a, g);
    }
}

fn lessStr(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

/// Order of the forms of one mnemonic (forms are tried in order, see vec.zig `pick`): legacy, VEX,
/// XOP, EVEX; by vector length; short `+r` forms first. For the stack, far-branch and SVM forms
/// the REX.W / 67 forms come first and the 16-bit operand-size (66) forms last, so an operand
/// that fits several (`push es`, `lcall [rax]`, `invlpga eax, ecx`) gets the default size.
fn rank(f: Form) u32 {
    const p: u32 = if (f.enc != .legacy or !has(&.{ "push", "pop", "lcall", "ljmp", "invlpga" }, f.mn)) 1 else if (f.rexw or std.mem.indexOfScalar(u8, f.pre, 0x67) != null) 0 else if (std.mem.indexOfScalar(u8, f.pre, 0x66) != null) 2 else 1;
    return @as(u32, @backingInt(f.enc)) * 64 + @as(u32, f.l) * 8 + p * 2 + @intFromBool(f.plus == null);
}

pub fn build(a: std.mem.Allocator, db: lj.Db) !std.StringArrayHashMapUnmanaged(std.ArrayList(Form)) {
    var names: std.ArrayList([]const u8) = .empty;
    for (db.instancesOf("X86Inst")) |nv| try names.append(a, nv.string);
    std.mem.sort([]const u8, names.items, {}, lessStr);

    var items: std.ArrayList(Item) = .empty;
    for (names.items) |n| {
        const r = db.get(n) orelse continue;
        if (lj.int(r, "isPseudo") != 0) continue;
        if (lj.int(r, "isCodeGenOnly") != 0 and lj.int(r, "ForceDisassemble") == 0) continue;
        if (lj.str(r, "AsmString").len == 0) continue;
        const f = fromRecord(a, n, r) catch |e| switch (e) {
            error.Skip => continue,
            else => return e,
        };
        if (eql(u8, f.mn, "wait")) {
            // `fwait` is the 9B prefix byte, encoded like the prefix of the wait forms below
            var g = f;
            g.pre = &.{0x9B};
            g.op = &.{};
            try items.append(a, .{ .rec = n, .f = g });
            continue;
        }
        try items.append(a, .{ .rec = n, .f = f });
        // `fstsw` etc. = `fwait` + `fnstsw` (the assembler's wait aliases)
        for (wait_forms) |wf| if (eql(u8, f.mn, wf[0])) {
            var g = f;
            g.mn = wf[1];
            g.pre = try std.mem.concat(a, u8, &.{ &.{0x9B}, f.pre });
            try items.append(a, .{ .rec = n, .f = g });
        };
    }
    try x87Aliases(a, db, &items);
    for (manual) |f| try items.append(a, .{ .rec = "", .f = f });

    var by: std.StringArrayHashMapUnmanaged(std.ArrayList(Form)) = .empty;
    for (items.items) |it| {
        var ex: std.ArrayList(Form) = .empty;
        try expandCond(a, it.f, &ex);
        const preds: []const []const u8 = if (db.get(it.rec)) |r| try predicates(a, r) else &.{};
        for (ex.items) |f0| {
            var f = f0;
            f.mn = recRename(it.rec) orelse rename(f.mn);
            if (has(&excluded, f.mn)) continue;
            if (has(&gp.mnemonics, f.mn) and !alsoHere(f)) continue;
            // selection hints (vec.zig `pick`): AVX10.2-only EVEX forms after the other EVEX forms;
            // the VEX form of a newer extension that also has an EVEX twin after the EVEX form
            var avx10 = false;
            var other = false;
            var late = false;
            for (preds) |p| {
                if (has(&.{ "In64BitMode", "Not64BitMode", "HasVLX", "HasEVEX512" }, p)) continue;
                if (eql(u8, p, "HasAVX10_2")) avx10 = true else if (!eql(u8, p, "HasMOVRS")) other = true;
                if (has(&.{ "HasAVXVNNI", "HasAVXIFMA", "HasAVXNECONVERT" }, p)) late = true;
            }
            if (avx10 and !other) f.lowprio = true;
            if (f.enc == .vex and late) f.late_vex = true;
            const gop = try by.getOrPut(a, f.mn);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(a, f);
        }
    }

    for (by.values()) |*lst| {
        // merge the masking / broadcast / rounding variants of one EVEX instruction
        var merged: std.ArrayList(Form) = .empty;
        var used = try a.alloc(bool, lst.items.len);
        @memset(used, false);
        for (lst.items, 0..) |f, i| {
            if (used[i]) continue;
            var group: std.ArrayList(Form) = .empty;
            for (lst.items[i..], i..) |g, j| if (!used[j] and sameGroup(f, g)) {
                used[j] = true;
                try group.append(a, g);
            };
            var base: ?Form = null;
            for (group.items) |g| if (g.bc == 0) {
                if (base == null or lessMask(g, base.?)) base = g;
            };
            const nb = base != null;
            var m = base orelse group.items[0];
            const ops = try a.dupe(Op, m.ops);
            for (group.items) |g| if (g.bc != 0) for (ops, g.ops) |*o, go| if (go.bcst != 0) {
                o.bcst = go.bcst;
                if (!nb) o.size = @as(u16, 16) << @intCast(m.l);
            };
            m.ops = ops;
            for (group.items) |g| {
                m.mask = @max(m.mask, g.mask);
                m.er = m.er or g.er;
                m.sae = m.sae or g.sae;
                m.lowprio = m.lowprio or g.lowprio;
            }
            try merged.append(a, m);
        }
        // er/sae and disp8*N belong to an instruction and vector length: copy between its register
        // and memory forms (only memory forms use n, only register forms use er/sae)
        for (merged.items, 0..) |*f, i| {
            if (f.enc != .evex) continue;
            if (isMem(f.*)) {
                for (merged.items, 0..) |g, j| if (j != i and sameEncoding(g, f.*) and !isMem(g)) {
                    if (g.er) f.er = true;
                    if (g.sae) f.sae = true;
                };
            } else {
                var n: u64 = 1;
                for (merged.items, 0..) |g, j| if (j != i and sameEncoding(g, f.*) and isMem(g)) {
                    n = g.n;
                    break;
                };
                f.n = n;
            }
        }
        // a memory form that is the r/m alternative of a GP-register form (`movd mm, r32/m32`) is
        // picked after the register forms
        for (merged.items, 0..) |*f, i| {
            const mi = for (f.ops, 0..) |o, k| {
                if (eql(u8, o.k, "mem")) break k;
            } else continue;
            for (merged.items, 0..) |g, j| {
                if (j == i or !sameEncoding(g, f.*) or g.ops.len != f.ops.len) continue;
                if (!eql(u8, g.ops[mi].k, "r32") and !eql(u8, g.ops[mi].k, "r64")) continue;
                var same = true;
                for (f.ops, g.ops, 0..) |p, q, k| if (k != mi and !opsEql(&.{p}, &.{q})) {
                    same = false;
                };
                if (same) f.lowprio = true;
            }
        }
        var any_evex = false;
        for (merged.items) |f| if (f.enc == .evex) {
            any_evex = true;
        };
        if (!any_evex) for (merged.items) |*f| {
            f.late_vex = false;
        };
        // the same form recorded for 32-bit and 64-bit mode (implicit eax/rax address): any mode
        var res: std.ArrayList(Form) = .empty;
        for (merged.items) |f| {
            if (!eql(u8, f.mode, "any")) {
                const other: []const u8 = if (eql(u8, f.mode, "x86")) "x64" else "x86";
                var fa = f;
                fa.mode = "any";
                const lf = try line(a, fa);
                var twin = false;
                for (merged.items) |g| if (eql(u8, g.mode, other)) {
                    var ga = g;
                    ga.mode = "any";
                    if (eql(u8, try line(a, ga), lf)) twin = true;
                };
                if (twin) {
                    if (eql(u8, f.mode, "x86")) try res.append(a, fa);
                    continue;
                }
            }
            try res.append(a, f);
        }
        const Ctx = struct {
            fn lt(_: void, x: Form, y: Form) bool {
                return rank(x) < rank(y);
            }
        };
        std.sort.insertion(Form, res.items, {}, Ctx.lt);
        lst.* = res;
    }
    return by;
}

fn lessMask(x: Form, y: Form) bool {
    if (x.mask != y.mask) return x.mask < y.mask;
    if (x.er != y.er) return !x.er;
    if (x.sae != y.sae) return !x.sae;
    return false;
}

/// x87 InstAlias records with the implied operand made explicit: `fxch` = `fxch st(1)`,
/// `faddp st(i)` = `faddp st(i), st`.
fn x87Aliases(a: std.mem.Allocator, db: lj.Db, items: *std.ArrayList(Item)) !void {
    var names: std.ArrayList([]const u8) = .empty;
    for (db.instancesOf("InstAlias")) |nv| try names.append(a, nv.string);
    std.mem.sort([]const u8, names.items, {}, lessStr);
    for (names.items) |an| {
        const al = db.get(an) orelse continue;
        const ri = al.get("ResultInst") orelse continue;
        if (ri != .object) continue;
        const pr = lj.str(ri.object, "printable");
        var it = std.mem.tokenizeAny(u8, std.mem.trim(u8, pr, "()"), " ");
        const rn = it.next() orelse continue;
        const arg = it.next();
        if (it.next() != null) continue;
        const r = db.get(rn) orelse continue;
        const ins = try lj.dagArgs(a, r, "InOperandList");
        const has_sti = for (ins) |x| {
            if (eql(u8, x.class, "RSTi")) break true;
        } else false;
        if (!has_sti or lj.str(r, "AsmString").len == 0) continue;
        if (lj.int(r, "isPseudo") != 0 or (lj.int(r, "isCodeGenOnly") != 0 and lj.int(r, "ForceDisassemble") == 0)) continue;
        const s = std.mem.trim(u8, try intel(a, lj.str(al, "AsmString"), false), " ");
        const tab = std.mem.indexOfScalar(u8, s, '\t') orelse s.len;
        const amn = std.mem.trim(u8, s[0..tab], " ");
        const rest = std.mem.trim(u8, s[tab..], " \t");
        var f = fromRecord(a, rn, r) catch |e| switch (e) {
            error.Skip => continue,
            else => return e,
        };
        if (!eql(u8, f.mn, amn)) continue;
        if (rest.len == 0 and arg != null and eql(u8, arg.?, "ST1")) {
            const op = try a.dupe(u8, f.op);
            op[op.len - 1] += 1;
            f.op = op;
            f.ops = &.{};
            f.plus = null;
        } else if (eql(u8, rest, "$op") and arg != null and eql(u8, arg.?, "RSTi:$op")) {
            f.ops = &.{.{ .k = "sti" }};
            f.plus = 0;
        } else continue;
        try items.append(a, .{ .rec = rn, .f = f });
    }
}

pub fn render(a: std.mem.Allocator, db: lj.Db, header: []const u8) ![]const u8 {
    const by = try build(a, db);
    const keys = try a.dupe([]const u8, by.keys());
    std.mem.sort([]const u8, keys, {}, lessStr);
    var n: usize = 0;
    for (by.values()) |v| n += v.items.len;
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s}// {d} forms, {d} mnemonics.\nconst Form = @import(\"vec.zig\").Form;\n\npub const Mnemonic = enum {{\n", .{ header, n, keys.len });
    for (keys) |k| try out.print(a, "    @\"{s}\",\n", .{k});
    try out.appendSlice(a, "};\n\npub const table = struct {\n");
    for (keys) |k| {
        try out.print(a, "    pub const @\"{s}\" = [_]Form{{\n", .{k});
        for (by.get(k).?.items) |f| try out.print(a, "        {s},\n", .{try line(a, f)});
        try out.appendSlice(a, "    };\n");
    }
    try out.appendSlice(a, "};\n");
    return out.items;
}

// ---- project decisions -----------------------------------------------------------------------

/// LLVM Intel spellings that differ from this table's mnemonics (the API keeps the names it had:
/// the AVX10.2 BF16 / FP8 names of the earlier ISE revisions, the x87 and string-op spellings).
fn rename(mn: []const u8) []const u8 {
    for (renames) |x| if (eql(u8, x[0], mn)) return x[1];
    return mn;
}
const renames = [_][2][]const u8{
    .{ "wait", "fwait" },                       .{ "fcompi", "fcomip" },                    .{ "fucompi", "fucomip" },
    .{ "popal", "popa" },                       .{ "pushal", "pusha" },                     .{ "popaw", "popa" },
    .{ "pushaw", "pusha" },                     .{ "tcvtrowps2bf16h", "tcvtrowps2pbf16h" }, .{ "tcvtrowps2bf16l", "tcvtrowps2pbf16l" },
    .{ "vcomisbf16", "vcomsbf16" },             .{ "vcmpbf16", "vcmppbf16" },               .{ "vaddbf16", "vaddnepbf16" },
    .{ "vsubbf16", "vsubnepbf16" },             .{ "vmulbf16", "vmulnepbf16" },             .{ "vdivbf16", "vdivnepbf16" },
    .{ "vsqrtbf16", "vsqrtnepbf16" },           .{ "vreducebf16", "vreducenepbf16" },       .{ "vrndscalebf16", "vrndscalenepbf16" },
    .{ "vminmaxbf16", "vminmaxnepbf16" },       .{ "vfpclassbf16", "vfpclasspbf16" },       .{ "vgetexpbf16", "vgetexppbf16" },
    .{ "vgetmantbf16", "vgetmantpbf16" },       .{ "vmaxbf16", "vmaxpbf16" },               .{ "vminbf16", "vminpbf16" },
    .{ "vrcpbf16", "vrcppbf16" },               .{ "vrsqrtbf16", "vrsqrtpbf16" },           .{ "vscalefbf16", "vscalefpbf16" },
    .{ "vfmadd132bf16", "vfmadd132nepbf16" },   .{ "vfmadd213bf16", "vfmadd213nepbf16" },   .{ "vfmadd231bf16", "vfmadd231nepbf16" },
    .{ "vfmsub132bf16", "vfmsub132nepbf16" },   .{ "vfmsub213bf16", "vfmsub213nepbf16" },   .{ "vfmsub231bf16", "vfmsub231nepbf16" },
    .{ "vfnmadd132bf16", "vfnmadd132nepbf16" }, .{ "vfnmadd213bf16", "vfnmadd213nepbf16" }, .{ "vfnmadd231bf16", "vfnmadd231nepbf16" },
    .{ "vfnmsub132bf16", "vfnmsub132nepbf16" }, .{ "vfnmsub213bf16", "vfnmsub213nepbf16" }, .{ "vfnmsub231bf16", "vfnmsub231nepbf16" },
    .{ "vcvt2ph2bf8", "vcvtne2ph2bf8" },        .{ "vcvt2ph2bf8s", "vcvtne2ph2bf8s" },      .{ "vcvt2ph2hf8", "vcvtne2ph2hf8" },
    .{ "vcvt2ph2hf8s", "vcvtne2ph2hf8s" },      .{ "vcvtph2bf8", "vcvtneph2bf8" },          .{ "vcvtph2bf8s", "vcvtneph2bf8s" },
    .{ "vcvtph2hf8", "vcvtneph2hf8" },          .{ "vcvtph2hf8s", "vcvtneph2hf8s" },        .{ "vcvtbf162ibs", "vcvtnebf162ibs" },
    .{ "vcvtbf162iubs", "vcvtnebf162iubs" },    .{ "lgdtd", "lgdt" },                       .{ "lidtd", "lidt" },
    .{ "sgdtd", "sgdt" },                       .{ "sidtd", "sidt" },                       .{ "vcvttbf162ibs", "vcvttnebf162ibs" },
    .{ "vcvttbf162iubs", "vcvttnebf162iubs" },
};

/// String instructions: one mnemonic with sized memory operands (`cmps m8, m8`), and `push imm16`
/// with the 66 prefix as `pushw`.
fn recRename(rec: []const u8) ?[]const u8 {
    if (eql(u8, rec, "PUSH16i")) return "pushw";
    const ops = [_][2][]const u8{ .{ "CMPS", "cmps" }, .{ "LODS", "lods" }, .{ "MOVS", "movs" }, .{ "SCAS", "scas" }, .{ "STOS", "stos" }, .{ "INS", "ins" }, .{ "OUTS", "outs" } };
    for (ops) |o| if (rec.len == o[0].len + 1 and std.mem.startsWith(u8, rec, o[0]) and std.mem.indexOfScalar(u8, "BWLQ", rec[rec.len - 1]) != null) return o[1];
    return null;
}

/// Mnemonics not offered: prefixes, Xeon Phi (AVX512ER/PF, 4FMAPS, 4VNNIW), VIA PadLock, Key
/// Locker wide/encode forms, SGX leaf functions, FRED, APX-only and 16-bit-only spellings, and
/// extensions added after the table was set up (AVX512BMM, the REX.W pcmpestr* spellings).
const excluded = [_][]const u8{
    "addr16",          "addr32",           "data16",          "rex64",           "lock",            "rep",
    "repne",           "xacquire",         "xrelease",        "cs",              "ds",              "es",
    "fs",              "gs",               "ss",              "v4fmaddps",       "v4fmaddss",       "v4fnmaddps",
    "v4fnmaddss",      "vp4dpwssd",        "vp4dpwssds",      "vexp2pd",         "vexp2ps",         "vrcp28pd",
    "vrcp28ps",        "vrcp28sd",         "vrcp28ss",        "vrsqrt28pd",      "vrsqrt28ps",      "vrsqrt28sd",
    "vrsqrt28ss",      "vgatherpf0dpd",    "vgatherpf0dps",   "vgatherpf0qpd",   "vgatherpf0qps",   "vgatherpf1dpd",
    "vgatherpf1dps",   "vgatherpf1qpd",    "vgatherpf1qps",   "vscatterpf0dpd",  "vscatterpf0dps",  "vscatterpf0qpd",
    "vscatterpf0qps",  "vscatterpf1dpd",   "vscatterpf1dps",  "vscatterpf1qpd",  "vscatterpf1qps",  "xcryptcbc",
    "xcryptcfb",       "xcryptctr",        "xcryptecb",       "xcryptofb",       "xsha1",           "xsha256",
    "xstore",          "montmul",          "aesdecwide128kl", "aesdecwide256kl", "aesencwide128kl", "aesencwide256kl",
    "encodekey128",    "encodekey256",     "encls",           "enclu",           "enclv",           "erets",
    "eretu",           "lkgs",             "jmpabs",          "retfq",           "popp",            "pushp",
    "salc",            "ffreep",           "ibhf",            "wrpkru",          "lgdtw",           "lidtw",
    "sgdtw",           "sidtw",            "pcmpestriq",      "pcmpestrmq",      "vpcmpestriq",     "vpcmpestrmq",
    "vbmacor16x16x16", "vbmacxor16x16x16", "vbitrevb",
};

/// GP-table mnemonics that also have forms here: the implicit-operand forms of clzero/monitor/
/// monitorx/pbndkb, and mov/push/pop with segment, control and debug registers.
fn alsoHere(f: Form) bool {
    if (has(&.{ "clzero", "monitor", "monitorx", "pbndkb" }, f.mn)) return true;
    if (has(&.{ "mov", "push", "pop" }, f.mn)) {
        for (f.ops) |o| if (has(&.{ "sreg", "creg", "dreg", "es", "cs", "ss", "ds", "fs", "gs" }, o.k)) return true;
    }
    return false;
}

const addr_in_reg = [_][]const u8{ "umonitor", "movdir64b", "enqcmd", "enqcmds" };

/// `f*` = `fwait` + `fn*`.
const wait_forms = [_][2][]const u8{
    .{ "fnclex", "fclex" }, .{ "fninit", "finit" },   .{ "fnsave", "fsave" },
    .{ "fnstcw", "fstcw" }, .{ "fnstenv", "fstenv" }, .{ "fnstsw", "fstsw" },
};

/// Forms LLVM 23 does not define.
const manual = [_]Form{
    // `invlpga eax, ecx` in 64-bit mode: 67 prefix (LLVM only has the rax form there)
    .{ .mn = "invlpga", .ops = &.{}, .pre = &.{0x67}, .map = 1, .op = &.{ 0x01, 0xDF }, .mode = "x64" },
    // `mov word ptr [m], sreg` / `mov sreg, word ptr [m]` with an operand-size prefix, as the
    // assembler encoded it before the LLVM tables (exact word size only; LLVM encodes no 66)
    .{ .mn = "mov", .ops = &.{ .{ .k = "mem", .size = 2 }, .{ .k = "sreg" } }, .pre = &.{0x66}, .op = &.{0x8C}, .reg = 1, .rm = 0, .exact_mem = true },
    .{ .mn = "mov", .ops = &.{ .{ .k = "sreg" }, .{ .k = "mem", .size = 2 } }, .pre = &.{0x66}, .op = &.{0x8E}, .reg = 0, .rm = 1, .exact_mem = true },
    // AMD APM: VMGEXIT  F2 0F 01 D9 (LLVM only accepts `rep vmmcall`)
    .{ .mn = "vmgexit", .ops = &.{}, .pre = &.{0xF2}, .map = 1, .op = &.{ 0x01, 0xD9 } },
    // Cyrix EMMI 3DNow! extensions: PFRCPV 0F 0F /r 86, PFRSQRTV 0F 0F /r 87
    .{ .mn = "pfrcpv", .ops = &.{ .{ .k = "mm" }, .{ .k = "mm" } }, .map = 4, .op = &.{0x86}, .reg = 0, .rm = 1 },
    .{ .mn = "pfrcpv", .ops = &.{ .{ .k = "mm" }, .{ .k = "mem", .size = 8 } }, .map = 4, .op = &.{0x86}, .reg = 0, .rm = 1 },
    .{ .mn = "pfrsqrtv", .ops = &.{ .{ .k = "mm" }, .{ .k = "mm" } }, .map = 4, .op = &.{0x87}, .reg = 0, .rm = 1 },
    .{ .mn = "pfrsqrtv", .ops = &.{ .{ .k = "mm" }, .{ .k = "mem", .size = 8 } }, .map = 4, .op = &.{0x87}, .reg = 0, .rm = 1 },
    // AMX-TF32 TMMULTF32PS tmm, tmm, tmm  VEX.128.66.0F38.W0 48 /r (removed from LLVM)
    .{ .mn = "tmmultf32ps", .ops = &.{ .{ .k = "tmm" }, .{ .k = "tmm" }, .{ .k = "tmm" } }, .enc = .vex, .map = 2, .op = &.{0x48}, .pp = 1, .w = 0, .reg = 0, .rm = 1, .vvvv = 2, .mode = "x64" },
};
