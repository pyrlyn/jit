//! x86-64 legacy-encoded general-purpose forms (src/x86_64/gp_table.zig) from LLVM's X86 target
//! description (`llvm-tblgen --dump-json X86.td`).
//!
//! Every LLVM instruction record with a legacy encoding is turned into concrete forms: operand
//! kinds in Intel operand order, prefixes, opcode bytes (map escape included), REX.W, ModRM role
//! of each operand, immediate size and CPU mode. Condition-code records (`cmov${cond}`,
//! `set${cond}`) are expanded into the 16 condition mnemonics. Which mnemonics belong to this table
//! and which CPU feature each needs are this project's own choices (see `mnemonics`, `features`).
const std = @import("std");
const lj = @import("llvm_json.zig");

pub const Form = struct {
    mn: []const u8,
    ops: []const []const u8,
    pre: []const u8,
    op: []const u8,
    rexw: bool,
    plus_r: bool,
    modrm: bool,
    digit: ?u3,
    imm: u8,
    reg: ?usize,
    rm: ?usize,
    opreg: ?usize,
    kind: Kind,
    features: []const []const u8,
    mode: []const u8, // "any", "x64", "x86"

    pub const Kind = enum { op, m, mr, rm };
};

/// Mnemonics this table provides (Intel names). Everything else is in vec_table or not offered.
pub const mnemonics = [_][]const u8{
    "aaa",        "aad",        "aam",         "aas",         "adc",         "adcx",        "add",         "adox",
    "and",        "arpl",       "bound",       "bsf",         "bsr",         "bswap",       "bt",          "btc",
    "btr",        "bts",        "cdq",         "cdqe",        "clc",         "cld",         "cldemote",    "clflush",
    "cli",        "clts",       "clui",        "clwb",        "clzero",      "cmc",         "cmovb",       "cmovbe",
    "cmovl",      "cmovle",     "cmovnb",      "cmovnbe",     "cmovnl",      "cmovnle",     "cmovno",      "cmovnp",
    "cmovns",     "cmovnz",     "cmovo",       "cmovp",       "cmovs",       "cmovz",       "cmp",         "cqo",
    "crc32",      "cwde",       "daa",         "das",         "div",         "endbr32",     "endbr64",     "fxrstor",
    "fxrstor64",  "fxsave",     "fxsave64",    "getsec",      "hlt",         "idiv",        "imul",        "in",
    "inc",        "int",        "int3",        "into",        "invd",        "invlpg",      "iretd",       "iretq",
    "jmp",        "lahf",       "lar",         "lea",         "leave",       "lfence",      "lsl",         "lzcnt",
    "mcommit",    "mfence",     "monitor",     "monitorx",    "mov",         "movabs",      "movbe",       "movdiri",
    "movnti",     "movsx",      "movsxd",      "movzx",       "mul",         "mwait",       "mwaitx",      "neg",
    "nop",        "not",        "or",          "out",         "pause",       "pbndkb",      "pconfig",     "pop",
    "popad",      "popcnt",     "popfd",       "popfq",       "prefetchit0", "prefetchit1", "prefetchnta", "prefetcht0",
    "prefetcht1", "prefetcht2", "prefetchw",   "prefetchwt1", "ptwrite",     "push",        "pushad",      "pushfd",
    "pushfq",     "rcl",        "rcr",         "rdfsbase",    "rdgsbase",    "rdmsr",       "rdpkru",      "rdpmc",
    "rdpru",      "rdtsc",      "rdtscp",      "ret",         "retf",        "rol",         "ror",         "rsm",
    "sahf",       "sar",        "saveprevssp", "sbb",         "serialize",   "setb",        "setbe",       "setl",
    "setle",      "setnb",      "setnbe",      "setnl",       "setnle",      "setno",       "setnp",       "setns",
    "setnz",      "seto",       "setp",        "sets",        "setssbsy",    "setz",        "sfence",      "shl",
    "shld",       "shr",        "shrd",        "stc",         "std",         "sti",         "stui",        "sub",
    "swapgs",     "syscall",    "sysenter",    "sysexit",     "sysexitq",    "sysret",      "sysretq",     "test",
    "testui",     "tzcnt",      "ud0",         "ud1",         "ud2",         "uiret",       "wbinvd",      "wbnoinvd",
    "wrfsbase",   "wrgsbase",   "wrmsr",       "wrmsrns",     "xchg",        "xend",        "xgetbv",      "xlatb",
    "xor",        "xresldtrk",  "xrstors",     "xrstors64",   "xsaves",      "xsaves64",    "xsetbv",      "xsusldtrk",
    "xtest",
};

/// CPU feature (isa.Feature) each mnemonic needs; baseline x86-64 otherwise.
const features = [_]struct { []const u8, []const []const u8 }{
    .{ "CET_IBT", &.{ "endbr32", "endbr64" } },
    .{ "CET_SS", &.{ "saveprevssp", "setssbsy" } },
    .{ "MSR", &.{ "rdmsr", "wrmsr" } },
    .{ "PCONFIG", &.{"pconfig"} },
    .{ "FXSR", &.{ "fxrstor", "fxrstor64", "fxsave", "fxsave64" } },
    .{ "PREFETCHI", &.{ "prefetchit0", "prefetchit1" } },
    .{ "PREFETCHWT1", &.{"prefetchwt1"} },
    .{ "MOVDIRI", &.{"movdiri"} },
    .{ "BMI", &.{"tzcnt"} },
    .{ "OSPKE", &.{"rdpkru"} },
    .{ "WRMSRNS", &.{"wrmsrns"} },
    .{ "SSE4_2", &.{"crc32"} },
    .{ "MCOMMIT", &.{"mcommit"} },
    .{ "RTM", &.{ "xend", "xtest" } },
    .{ "LZCNT", &.{"lzcnt"} },
    .{ "XSAVES", &.{ "xrstors", "xrstors64", "xsaves", "xsaves64" } },
    .{ "TSXLDTRK", &.{ "xresldtrk", "xsusldtrk" } },
    .{ "RDPRU", &.{"rdpru"} },
    .{ "MOVBE", &.{"movbe"} },
    .{ "SMX", &.{"getsec"} },
    .{ "PTWRITE", &.{"ptwrite"} },
    .{ "CMOV", &.{ "cmovb", "cmovbe", "cmovl", "cmovle", "cmovnb", "cmovnbe", "cmovnl", "cmovnle", "cmovno", "cmovnp", "cmovns", "cmovnz", "cmovo", "cmovp", "cmovs", "cmovz" } },
    .{ "CLFLUSH", &.{"clflush"} },
    .{ "XSAVE", &.{ "xgetbv", "xsetbv" } },
    .{ "WBNOINVD", &.{"wbnoinvd"} },
    .{ "RDTSCP", &.{"rdtscp"} },
    .{ "POPCNT", &.{"popcnt"} },
    .{ "UINTR", &.{ "clui", "stui", "testui", "uiret" } },
    .{ "SSE2", &.{ "lfence", "mfence", "movnti" } },
    .{ "I486", &.{ "invd", "invlpg", "wbinvd" } },
    .{ "LAHFSAHF", &.{ "lahf", "sahf" } },
    .{ "ADX", &.{ "adcx", "adox" } },
    .{ "RDTSC", &.{"rdtsc"} },
    .{ "TSE", &.{"pbndkb"} },
    .{ "PREFETCHW", &.{"prefetchw"} },
    .{ "MONITORX", &.{ "monitorx", "mwaitx" } },
    .{ "CLZERO", &.{"clzero"} },
    .{ "CLDEMOTE", &.{"cldemote"} },
    .{ "FSGSBASE", &.{ "rdfsbase", "rdgsbase", "wrfsbase", "wrgsbase" } },
    .{ "SSE", &.{ "prefetchnta", "prefetcht0", "prefetcht1", "prefetcht2", "sfence" } },
    .{ "CLWB", &.{"clwb"} },
    .{ "SERIALIZE", &.{"serialize"} },
    .{ "MONITOR", &.{ "monitor", "mwait" } },
};

fn featuresOf(a: std.mem.Allocator, mn: []const u8) ![]const []const u8 {
    for (features) |f| for (f[1]) |m| if (std.mem.eql(u8, m, mn)) return a.dupe([]const u8, &.{f[0]});
    return &.{};
}

/// Condition-code suffixes in encoding order (`cc` added to the opcode).
pub const cc_names = [16][]const u8{ "o", "no", "b", "nb", "z", "nz", "be", "nbe", "s", "ns", "p", "np", "l", "nl", "le", "nle" };

fn inList(mn: []const u8) bool {
    for (mnemonics) |m| if (std.mem.eql(u8, m, mn)) return true;
    return false;
}

/// LLVM operand class -> table operand kind.
fn opKind(class: []const u8) ?[]const u8 {
    const map = [_][2][]const u8{
        .{ "GR8", "r8" },                 .{ "GR8_NOREX", "r8" },   .{ "GR16", "r16" },       .{ "GR32", "r32" },
        .{ "GR64", "r64" },               .{ "i8mem", "m8" },       .{ "i16mem", "m16" },     .{ "i32mem", "m32" },
        .{ "i64mem", "m64" },             .{ "anymem", "m" },       .{ "opaquemem", "m" },    .{ "lea64mem", "m" },
        .{ "lea64_32mem", "m" },          .{ "i8imm", "i8" },       .{ "u8imm", "i8" },       .{ "i16imm", "i16" },
        .{ "i32imm", "i32" },             .{ "i64imm", "i64" },     .{ "i16i8imm", "s8" },    .{ "i32i8imm", "s8" },
        .{ "i64i8imm", "s8" },            .{ "i64i32imm", "s32" },  .{ "i16u8imm", "i8" },    .{ "i32u8imm", "i8" },
        .{ "i64u8imm", "i8" },            .{ "i8mem_NOREX", "m8" }, .{ "GR32_NOREX", "r32" }, .{ "u16imm", "u16" },
        .{ "GR16orGR32orGR64", "rsame" },
    };
    for (map) |m| if (std.mem.eql(u8, m[0], class)) return m[1];
    return null;
}

const imm_kinds = [_][]const u8{ "i8", "s8", "u8", "i16", "u16", "i32", "s32", "u32", "i64" };
fn isImm(k: []const u8) bool {
    for (imm_kinds) |x| if (std.mem.eql(u8, x, k)) return true;
    return false;
}
const fixed_regs = [_][]const u8{ "al", "ax", "eax", "rax", "cl", "dx" };
fn isFixed(k: []const u8) bool {
    for (fixed_regs) |x| if (std.mem.eql(u8, x, k)) return true;
    return std.mem.eql(u8, k, "one");
}

const Skip = error{Skip};

/// All forms from one LLVM record (several for condition-code records).
fn fromRecord(a: std.mem.Allocator, rname: []const u8, r: lj.Obj, out: *std.ArrayList(Form)) !void {
    // far jumps/calls (segment:offset) are not offered
    if (std.mem.startsWith(u8, rname, "FAR")) return;
    // code-generator-only records that the assembler/disassembler do not use (`*Locked`, ...)
    if (lj.int(r, "isCodeGenOnly") != 0 and lj.int(r, "ForceDisassemble") == 0) return;
    if (!std.mem.eql(u8, lj.def(r, "OpEnc"), "EncNormal")) return;
    if (lj.int(r, "isPseudo") != 0) return;
    if (!std.mem.eql(u8, lj.def(r, "AdSize"), "AdSizeX")) return;
    const asm_s = try lj.asmVariant(a, lj.str(r, "AsmString"), 1);
    const form = lj.def(r, "Form");
    const immt = lj.def(r, "ImmT");
    if (std.mem.indexOf(u8, immt, "PCRel") != null) return;
    const trimmed = std.mem.trim(u8, asm_s, " \t");
    if (trimmed.len == 0) return;
    const sp = std.mem.indexOfAny(u8, trimmed, " \t") orelse trimmed.len;
    const mn0 = trimmed[0..sp];
    const rest = std.mem.trim(u8, trimmed[sp..], " \t");

    const ins = try lj.dagArgs(a, r, "InOperandList");
    const outs = try lj.dagArgs(a, r, "OutOperandList");
    const tied = try lj.tiedPairs(a, r);
    // MI operands in order (outs, then ins not tied to an out).
    var mi: std.ArrayList(lj.DagArg) = .empty;
    for (outs) |o| try mi.append(a, o);
    ins_loop: for (ins) |o| {
        if (std.mem.eql(u8, o.class, "variable_ops")) continue;
        for (tied) |t| if (std.mem.eql(u8, t[0], o.name) or std.mem.eql(u8, t[1], o.name)) {
            for (outs) |x| if (std.mem.eql(u8, x.name, t[0]) or std.mem.eql(u8, x.name, t[1])) continue :ins_loop;
        };
        if (std.mem.eql(u8, o.class, "ccode")) continue;
        try mi.append(a, o);
    }
    const canon = struct {
        fn f(name: []const u8, tp: [][2][]const u8) []const u8 {
            // map a tied name onto its partner that comes first in `outs`
            for (tp) |t| if (std.mem.eql(u8, t[0], name) or std.mem.eql(u8, t[1], name)) return t[1];
            return name;
        }
    }.f;

    // Asm operands (Intel order).
    var ops: std.ArrayList([]const u8) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    if (rest.len != 0) {
        var it = std.mem.splitScalar(u8, rest, ',');
        while (it.next()) |tok0| {
            const tok = std.mem.trim(u8, tok0, " \t");
            if (tok.len == 0) return;
            if (tok[0] == '$') {
                var nm = tok[1..];
                if (nm.len > 0 and nm[0] == '{') nm = nm[1 .. std.mem.indexOfScalar(u8, nm, '}') orelse return];
                var cls: ?[]const u8 = null;
                for (ins) |x| if (std.mem.eql(u8, x.name, nm)) {
                    cls = x.class;
                };
                for (outs) |x| if (std.mem.eql(u8, x.name, nm)) {
                    cls = x.class;
                };
                const k = opKind(cls orelse return) orelse return;
                try ops.append(a, k);
                try names.append(a, canon(nm, tied));
            } else {
                const t = if (tok[0] == '%') tok[1..] else tok;
                if (!isFixed(t)) return;
                try ops.append(a, t);
                try names.append(a, "");
            }
        }
    }

    var opbytes: std.ArrayList(u8) = .empty;
    const map = lj.def(r, "OpMap");
    if (std.mem.eql(u8, map, "TB")) {
        try opbytes.append(a, 0x0f);
    } else if (std.mem.eql(u8, map, "T8")) {
        try opbytes.appendSlice(a, &.{ 0x0f, 0x38 });
    } else if (std.mem.eql(u8, map, "TA")) {
        try opbytes.appendSlice(a, &.{ 0x0f, 0x3a });
    } else if (!std.mem.eql(u8, map, "OB")) return;
    const opcode: u8 = @intCast(lj.bits(r, "Opcode"));
    // 82 /digit: 32-bit-only alias of 80 /digit (LLVM keeps it for the disassembler)
    if (opcode == 0x82 and std.mem.eql(u8, map, "OB")) return;

    var pre: std.ArrayList(u8) = .empty;
    const opsize = lj.def(r, "OpSize");
    if (std.mem.eql(u8, opsize, "OpSize16")) try pre.append(a, 0x66);
    const pfx = lj.def(r, "OpPrefix");
    if (std.mem.eql(u8, pfx, "PD")) {
        try pre.append(a, 0x66);
    } else if (std.mem.eql(u8, pfx, "XS")) {
        try pre.append(a, 0xf3);
    } else if (std.mem.eql(u8, pfx, "XD")) {
        try pre.append(a, 0xf2);
    } else if (!std.mem.eql(u8, pfx, "NoPrfx") and !std.mem.eql(u8, pfx, "PS")) return;

    var fm = Form{
        .mn = mn0,
        .ops = ops.items,
        .pre = pre.items,
        .op = &.{},
        .rexw = lj.int(r, "hasREX_W") != 0,
        .plus_r = false,
        .modrm = false,
        .digit = null,
        .imm = 0,
        .reg = null,
        .rm = null,
        .opreg = null,
        .kind = .op,
        .features = &.{},
        .mode = "any",
    };
    fm.imm = if (std.mem.eql(u8, immt, "NoImm")) 0 else if (std.mem.eql(u8, immt, "Imm8")) 1 else if (std.mem.eql(u8, immt, "Imm16")) 2 else if (std.mem.eql(u8, immt, "Imm32") or std.mem.eql(u8, immt, "Imm32S")) 4 else if (std.mem.eql(u8, immt, "Imm64")) 8 else return;

    const pos = struct {
        fn f(nms: []const []const u8, n: []const u8, tp: [][2][]const u8) ?usize {
            const c = canon(n, tp);
            for (nms, 0..) |x, i| if (std.mem.eql(u8, x, c)) return i;
            return null;
        }
    }.f;
    const is_cc = std.mem.endsWith(u8, form, "CC");
    const base_form = if (is_cc) form[0 .. form.len - 2] else form;
    var extra: ?u8 = null;
    if (std.mem.eql(u8, base_form, "MRMDestReg") or std.mem.eql(u8, base_form, "MRMDestMem")) {
        if (mi.items.len < 2) return;
        fm.kind = .mr;
        fm.modrm = true;
        fm.rm = pos(names.items, mi.items[0].name, tied) orelse return;
        fm.reg = pos(names.items, mi.items[1].name, tied) orelse return;
    } else if (std.mem.eql(u8, base_form, "MRMSrcReg") or std.mem.eql(u8, base_form, "MRMSrcMem")) {
        if (mi.items.len < 2) return;
        fm.kind = .rm;
        fm.modrm = true;
        fm.reg = pos(names.items, mi.items[0].name, tied) orelse return;
        fm.rm = pos(names.items, mi.items[1].name, tied) orelse return;
    } else if (base_form.len == 5 and std.mem.startsWith(u8, base_form, "MRM") and (base_form[4] == 'r' or base_form[4] == 'm')) {
        if (mi.items.len < 1) return;
        fm.kind = .m;
        fm.modrm = true;
        fm.digit = if (base_form[3] == 'X') 0 else if (base_form[3] >= '0' and base_form[3] <= '7') @intCast(base_form[3] - '0') else return;
        fm.rm = pos(names.items, mi.items[0].name, tied) orelse return;
    } else if (std.mem.eql(u8, base_form, "AddRegFrm")) {
        if (mi.items.len < 1) return;
        fm.plus_r = true;
        fm.opreg = pos(names.items, mi.items[0].name, tied) orelse return;
    } else if (std.mem.eql(u8, base_form, "RawFrm") or std.mem.eql(u8, base_form, "RawFrmImm8")) {
        // no register operands
    } else if (base_form.len == 5 and std.mem.startsWith(u8, base_form, "MRM") and base_form[4] == 'X' and base_form[3] >= '0' and base_form[3] <= '7') {
        extra = 0xc0 | ((base_form[3] - '0') << 3); // fixed ModRM byte, mod=11 rm=0
    } else if (std.mem.startsWith(u8, base_form, "MRM_")) {
        extra = std.fmt.parseInt(u8, base_form[4..], 16) catch return;
    } else return;
    // lar/lsl: the selector register is written with the destination's width (r32 for r64)
    for (ops.items) |*k| if (std.mem.eql(u8, k.*, "rsame")) {
        k.* = "r16";
        for (ops.items) |o| if (o[0] == 'r' and !std.mem.eql(u8, o, "rsame")) {
            k.* = if (std.mem.eql(u8, o, "r64")) "r32" else o;
        };
    };
    // shift/rotate by one: implicit `1` operand
    if ((opcode == 0xd0 or opcode == 0xd1) and fm.kind == .m and fm.imm == 0) try ops.append(a, "one");
    fm.ops = ops.items;

    // every explicit non-fixed, non-immediate operand must have a ModRM/opcode role
    var nimm: usize = 0;
    for (fm.ops, 0..) |k, i| {
        if (isImm(k)) {
            nimm += 1;
            continue;
        }
        if (isFixed(k)) continue;
        if (fm.reg != i and fm.rm != i and fm.opreg != i) return;
    }
    if (nimm != @as(usize, if (fm.imm != 0) 1 else 0) and !std.mem.eql(u8, base_form, "RawFrmImm8")) return;
    if (std.mem.eql(u8, base_form, "RawFrmImm8")) return; // enter: two immediates (not in this table)

    const mode: []const u8 = if (lj.hasDef(r, "Predicates", "In64BitMode")) "x64" else if (lj.hasDef(r, "Predicates", "Not64BitMode")) "x86" else "any";
    fm.mode = mode;

    const n_cc: usize = if (is_cc) 16 else 1;
    for (0..n_cc) |cc| {
        var g = fm;
        var bytes = try opbytes.clone(a);
        try bytes.append(a, opcode + @as(u8, @intCast(cc)));
        if (extra) |e| try bytes.append(a, e);
        g.op = bytes.items;
        if (is_cc) {
            const at = std.mem.indexOf(u8, mn0, "${cond}") orelse return;
            g.mn = try std.mem.concat(a, u8, &.{ mn0[0..at], cc_names[cc], mn0[at + 7 ..] });
        }
        g.mn = rename(g.mn);
        if (!inList(g.mn)) continue;
        g.features = try featuresOf(a, g.mn);
        try out.append(a, g);
    }
}

/// LLVM Intel mnemonic spellings that differ from ours.
fn rename(mn: []const u8) []const u8 {
    const m = [_][2][]const u8{ .{ "retfq", "retf" }, .{ "lretq", "retf" }, .{ "popal", "popad" }, .{ "pushal", "pushad" } };
    for (m) |x| if (std.mem.eql(u8, x[0], mn)) return x[1];
    return mn;
}

fn kindRank(k: Form.Kind) u8 {
    return @backingInt(k);
}

pub fn forms(a: std.mem.Allocator, db: lj.Db) !std.StringArrayHashMapUnmanaged(std.ArrayList(Form)) {
    var all: std.ArrayList(Form) = .empty;
    for (db.instancesOf("X86Inst")) |nv| {
        const r = db.get(nv.string) orelse continue;
        try fromRecord(a, nv.string, r, &all);
    }
    try fixups(a, &all);
    var by: std.StringArrayHashMapUnmanaged(std.ArrayList(Form)) = .empty;
    for (all.items) |f| {
        const gop = try by.getOrPut(a, f.mn);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        // de-dup
        var dup = false;
        for (gop.value_ptr.items) |g| if (try eqlForm(a, f, g)) {
            dup = true;
            break;
        };
        if (!dup) try gop.value_ptr.append(a, f);
    }
    return by;
}

/// Memory operand sizes where LLVM records the element size but the instruction reads more.
const mem_size = [_]struct { []const u8, []const u8, []const u8 }{
    .{ "bound", "m16", "m32" }, // m16&16
    .{ "bound", "m32", "m64" }, // m32&32
    .{ "invlpg", "m8", "m" }, // a linear address, no access size
};

/// Cache-control / prefetch instructions take an address, not a sized access.
const untyped_mem = [_][]const u8{ "cldemote", "clflush", "clwb", "prefetchit0", "prefetchit1", "prefetchnta", "prefetcht0", "prefetcht1", "prefetcht2", "prefetchw", "prefetchwt1" };

/// Forms LLVM records that this table does not offer (operand-size variants for 16-bit code and
/// assembler-only duplicates).
fn excluded(f: Form) bool {
    const eql = std.mem.eql;
    const has66 = f.pre.len > 0 and f.pre[0] == 0x66;
    // 16-bit near/far returns, 16-bit indirect jmp and `pushw imm`: 16-bit code only
    if ((eql(u8, f.mn, "ret") or eql(u8, f.mn, "retf") or eql(u8, f.mn, "jmp")) and has66) return true;
    if (eql(u8, f.mn, "retf") and f.rexw) return true;
    if (eql(u8, f.mn, "push") and has66 and f.ops.len == 1 and isImm(f.ops[0])) return true;
    // movsx/movzx between registers of the same width (LLVM disassembler-only)
    if ((eql(u8, f.mn, "movsx") or eql(u8, f.mn, "movzx")) and f.ops.len == 2 and eql(u8, f.ops[0][1..], f.ops[1][1..])) return true;
    // ud1: SDM defines only `ud1 r32, r/m32`
    if (eql(u8, f.mn, "ud1") and !eql(u8, f.ops[0], "r32")) return true;
    return false;
}

/// Forms not in LLVM's X86 description: instructions it does not define (mcommit, ud0), and
/// assembler conveniences of this project (`nop r/m, r` multi-byte NOP with a register field,
/// `and r64, imm32` as a zero-extending 32-bit AND, `mov r64, imm64` = movabs).
fn manual(a: std.mem.Allocator, all: *std.ArrayList(Form)) !void {
    const base = Form{ .mn = "", .ops = &.{}, .pre = &.{}, .op = &.{}, .rexw = false, .plus_r = false, .modrm = false, .digit = null, .imm = 0, .reg = null, .rm = null, .opreg = null, .kind = .op, .features = &.{}, .mode = "any" };
    var f = base;
    // AMD APM: MCOMMIT  F3 0F 01 FA
    f.mn = "mcommit";
    f.pre = &.{0xf3};
    f.op = &.{ 0x0f, 0x01, 0xfa };
    f.features = &.{"MCOMMIT"};
    try all.append(a, f);
    // Intel SDM: UD0 r32, r/m32  0F FF /r
    for ([_][]const u8{ "r32", "m32" }) |k| {
        f = base;
        f.mn = "ud0";
        f.ops = try a.dupe([]const u8, &.{ "r32", k });
        f.op = &.{ 0x0f, 0xff };
        f.modrm = true;
        f.reg = 0;
        f.rm = 1;
        f.kind = .rm;
        try all.append(a, f);
    }
    // nop r/m, r  (0F 1F /r)
    for ([_]struct { []const u8, []const u8, []const u8, bool }{
        .{ "r16", "m16", "r16", false }, .{ "r32", "m32", "r32", false }, .{ "r64", "m64", "r64", true },
    }) |sz| for ([_][]const u8{ sz[0], sz[1] }) |rmk| {
        f = base;
        f.mn = "nop";
        f.ops = try a.dupe([]const u8, &.{ rmk, sz[2] });
        f.pre = if (std.mem.eql(u8, sz[2], "r16")) &.{0x66} else &.{};
        f.op = &.{ 0x0f, 0x1f };
        f.rexw = sz[3];
        f.modrm = true;
        f.reg = 1;
        f.rm = 0;
        f.kind = .mr;
        try all.append(a, f);
    };
    // and r64, u32 / and rax, u32: 32-bit AND (zero-extends, clears the upper half)
    f = base;
    f.mn = "and";
    f.ops = &.{ "r64", "u32" };
    f.op = &.{0x81};
    f.modrm = true;
    f.digit = 4;
    f.imm = 4;
    f.rm = 0;
    f.kind = .m;
    try all.append(a, f);
    f = base;
    f.mn = "and";
    f.ops = &.{ "rax", "u32" };
    f.op = &.{0x25};
    f.imm = 4;
    try all.append(a, f);
}

fn fixups(a: std.mem.Allocator, all: *std.ArrayList(Form)) !void {
    try manual(a, all);
    {
        var i: usize = 0;
        while (i < all.items.len) {
            if (excluded(all.items[i])) _ = all.orderedRemove(i) else i += 1;
        }
    }
    const n0 = all.items.len;
    for (0..n0) |idx| {
        const f = all.items[idx];
        // `mov r64, imm64` is also spelled movabs
        if (std.mem.eql(u8, f.mn, "movabs") and f.ops.len == 2 and std.mem.eql(u8, f.ops[1], "i64") and f.plus_r) {
            var g = f;
            g.mn = "mov";
            try all.append(a, g);
        }
        // xchg is symmetric: offer both operand orders
        if (std.mem.eql(u8, f.mn, "xchg") and f.ops.len == 2) {
            var g = f;
            g.ops = try a.dupe([]const u8, &.{ f.ops[1], f.ops[0] });
            if (f.modrm) {
                g.reg = f.rm;
                g.rm = f.reg;
                g.kind = if (f.kind == .rm) .mr else .rm;
            } else if (f.opreg) |o| g.opreg = 1 - o;
            try all.append(a, g);
        }
    }
    for (all.items) |*f| {
        // push imm8 (6A) accepts any 8-bit value (see gp.zig)
        if (std.mem.eql(u8, f.mn, "push") and f.op.len == 1 and f.op[0] == 0x6a) f.ops = &.{"i8"};
        // movsxd r16, r/m16 (SDM: 66 63 /r); LLVM spells the source as 32-bit
        if (std.mem.eql(u8, f.mn, "movsxd") and std.mem.eql(u8, f.ops[0], "r16")) {
            f.ops = try a.dupe([]const u8, &.{ "r16", if (f.ops[1][0] == 'r') "r16" else "m16" });
        }
        for (untyped_mem) |m| if (std.mem.eql(u8, f.mn, m)) {
            const ops = try a.dupe([]const u8, f.ops);
            for (ops) |*k| if (k.*[0] == 'm') {
                k.* = "m";
            };
            f.ops = ops;
        };
        for (mem_size) |m| if (std.mem.eql(u8, f.mn, m[0])) {
            const ops = try a.dupe([]const u8, f.ops);
            for (ops) |*k| if (std.mem.eql(u8, k.*, m[1])) {
                k.* = m[2];
                break;
            };
            if (!std.mem.eql(u8, ops[0], f.ops[0]) or !std.mem.eql(u8, ops[ops.len - 1], f.ops[f.ops.len - 1])) {
                f.ops = ops;
                break;
            }
        };
        // `ret imm16` / `retf imm16`: unsigned stack adjustment
        if (std.mem.eql(u8, f.mn, "ret") or std.mem.eql(u8, f.mn, "retf")) {
            const ops = try a.dupe([]const u8, f.ops);
            for (ops) |*k| if (std.mem.eql(u8, k.*, "i16")) {
                k.* = "u16";
            };
            f.ops = ops;
        }
    }
    // Same encoding recorded once per mode (LEA32r / LEA64_32r, LEAVE / LEAVE64): any mode.
    var i: usize = 0;
    while (i < all.items.len) : (i += 1) {
        const f = &all.items[i];
        if (std.mem.eql(u8, f.mode, "any")) continue;
        var j = i + 1;
        while (j < all.items.len) : (j += 1) {
            const g = all.items[j];
            if (!std.mem.eql(u8, g.mn, f.mn) or std.mem.eql(u8, g.mode, f.mode) or std.mem.eql(u8, g.mode, "any")) continue;
            var x = f.*;
            x.mode = g.mode;
            if (try eqlForm(a, x, g)) {
                f.mode = "any";
                _ = all.orderedRemove(j);
                break;
            }
        }
    }
    // A REX.W form exists only in 64-bit mode anyway; mark it `x64` only when the mnemonic itself is
    // 64-bit-only (has a 64-bit-only form without REX.W, e.g. rdfsbase r32).
    for (all.items) |*f| {
        if (!f.rexw or !std.mem.eql(u8, f.mode, "x64")) continue;
        var only64 = false;
        for (all.items) |g| if (std.mem.eql(u8, g.mn, f.mn) and !g.rexw and std.mem.eql(u8, g.mode, "x64")) {
            only64 = true;
        };
        if (!only64) f.mode = "any";
    }
    // SDM: #UD outside 64-bit mode (LLVM does not restrict these)
    for (all.items) |*f| for ([_][]const u8{ "swapgs", "syscall", "sysret", "sysretq" }) |m| if (std.mem.eql(u8, f.mn, m)) {
        f.mode = "x64";
    };
}

fn eqlForm(a: std.mem.Allocator, x: Form, y: Form) !bool {
    return std.mem.eql(u8, try line(a, x), try line(a, y));
}

fn optStr(a: std.mem.Allocator, v: anytype) ![]const u8 {
    return if (v) |x| try std.fmt.allocPrint(a, "{d}", .{x}) else "null";
}

fn hexList(a: std.mem.Allocator, bs: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (bs, 0..) |b, i| {
        if (i != 0) try out.appendSlice(a, ", ");
        try out.print(a, "0x{x}", .{b});
    }
    return out.items;
}

pub fn line(a: std.mem.Allocator, f: Form) ![]const u8 {
    var ops: std.ArrayList(u8) = .empty;
    for (f.ops, 0..) |k, i| {
        if (i != 0) try ops.appendSlice(a, ", ");
        try ops.print(a, ".{s}", .{k});
    }
    var feats: std.ArrayList(u8) = .empty;
    for (f.features, 0..) |x, i| {
        if (i != 0) try feats.appendSlice(a, ", ");
        if (std.ascii.isDigit(x[0])) try feats.print(a, ".@\"{s}\"", .{x}) else try feats.print(a, ".{s}", .{x});
    }
    const mode = if (std.mem.eql(u8, f.mode, "any")) "" else try std.fmt.allocPrint(a, ", .mode = .{s}", .{f.mode});
    return std.fmt.allocPrint(a, ".{{ .ops = &.{{{s}}}, .pre = &.{{{s}}}, .op = &.{{{s}}}, .rexw = {}, .plus_r = {}, .modrm = {}, .digit = {s}, .imm = {d}, .reg = {s}, .rm = {s}, .opreg = {s}, .features = &.{{{s}}}{s} }}", .{
        ops.items,              try hexList(a, f.pre), try hexList(a, f.op), f.rexw,              f.plus_r,               f.modrm,
        try optStr(a, f.digit), f.imm,                 try optStr(a, f.reg), try optStr(a, f.rm), try optStr(a, f.opreg), feats.items,
        mode,
    });
}

const zig_keywords = [_][]const u8{ "and", "or", "test", "int", "out", "in" };
fn zname(a: std.mem.Allocator, n: []const u8) ![]const u8 {
    for (zig_keywords) |k| if (std.mem.eql(u8, k, n)) return std.fmt.allocPrint(a, "@\"{s}\"", .{n});
    return n;
}

fn lessStr(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

pub const Output = struct { zig: []const u8, json: []const u8 };

/// `gp_table.zig` plus `scripts/.gp_forms.json` (the forms as JSON, read by scripts/gen_gp_tests.py
/// and scripts/fuzz_gp.py).
pub fn render(a: std.mem.Allocator, db: lj.Db, header: []const u8) !Output {
    var by = try forms(a, db);
    // reg,reg ties: LLVM's assembler picks MR (MRMDestReg) over RM, except xchg (RM).
    for (by.keys(), by.values()) |mn, *lst| {
        const first: Form.Kind = if (std.mem.eql(u8, mn, "xchg")) .rm else .mr;
        const Ctx = struct {
            first: Form.Kind,
            fn lt(c: @This(), x: Form, y: Form) bool {
                const rx: u8 = if (x.kind == c.first) 0 else 1;
                const ry: u8 = if (y.kind == c.first) 0 else 1;
                return rx < ry;
            }
        };
        std.sort.insertion(Form, lst.items, Ctx{ .first = first }, Ctx.lt);
    }
    const keys = try a.dupe([]const u8, by.keys());
    std.mem.sort([]const u8, keys, {}, lessStr);
    var n: usize = 0;
    for (by.values()) |v| n += v.items.len;
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{s}// {d} forms, {d} mnemonics.\nconst Form = @import(\"gp.zig\").Form;\n\npub const Mnemonic = enum {{\n", .{ header, n, keys.len });
    for (keys) |k| try out.print(a, "    {s},\n", .{try zname(a, k)});
    try out.appendSlice(a, "};\n\npub const table = struct {\n");
    for (keys) |k| {
        try out.print(a, "    pub const {s} = [_]Form{{\n", .{try zname(a, k)});
        for (by.get(k).?.items) |f| try out.print(a, "        {s},\n", .{try line(a, f)});
        try out.appendSlice(a, "    };\n");
    }
    try out.appendSlice(a, "};\n");

    var js: std.ArrayList(u8) = .empty;
    try js.append(a, '{');
    for (keys, 0..) |k, ki| {
        if (ki != 0) try js.appendSlice(a, ", ");
        try js.print(a, "\"{s}\": [", .{k});
        for (by.get(k).?.items, 0..) |f, fi| {
            if (fi != 0) try js.appendSlice(a, ", ");
            try js.appendSlice(a, "{\"ops\": [");
            for (f.ops, 0..) |o, i| try js.print(a, "{s}\"{s}\"", .{ if (i != 0) ", " else "", o });
            try js.appendSlice(a, "], \"pre\": [");
            for (f.pre, 0..) |o, i| try js.print(a, "{s}{d}", .{ if (i != 0) ", " else "", o });
            try js.appendSlice(a, "], \"op\": [");
            for (f.op, 0..) |o, i| try js.print(a, "{s}{d}", .{ if (i != 0) ", " else "", o });
            try js.print(a, "], \"rexw\": {}, \"plus\": {}, \"modrm\": {}, \"digit\": {s}, \"imm\": {d}, \"reg\": {s}, \"rm\": {s}, \"opreg\": {s}, \"kind\": \"{s}\", \"mode\": \"{s}\", \"ext\": [", .{
                f.rexw, f.plus_r, f.modrm, try optStr(a, f.digit), f.imm, try optStr(a, f.reg), try optStr(a, f.rm), try optStr(a, f.opreg), @tagName(f.kind), f.mode,
            });
            for (f.features, 0..) |o, i| try js.print(a, "{s}\"{s}\"", .{ if (i != 0) ", " else "", o });
            try js.appendSlice(a, "]}");
        }
        try js.append(a, ']');
    }
    try js.appendSlice(a, "}\n");
    return .{ .zig = out.items, .json = js.items };
}
