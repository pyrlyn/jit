//! Validation tables `src/x86_64/isa.zig` and `src/aarch64/isa.zig` from LLVM's target
//! descriptions (`llvm-tblgen --dump-json` of X86.td / AArch64.td).
//!
//! Each table lists, for every instruction the hand-written encoders emit, the LLVM record(s) that
//! define its encoding and the facts read from them:
//!
//! - x86-64: opcode bytes (`OpMap` + `Opcode`, condition code added for `AddCCFrm`), opcode-register
//!   form (`AddRegFrm`), ModRM use and `/digit` (`Form`), immediate kind (`ImmT`), CPU features
//!   (`Predicates` that name a `SubtargetFeature`).
//! - AArch64: fixed bits (`Inst`, with the fixed operands of an `InstAlias` applied: XZR -> 31,
//!   literal -> value), whether Rd/Rn accept SP (operand class `GPR64sp`), immediate field width
//!   and scale (operand class), CPU features (`Predicates`).
//!
//! Project decisions that are not LLVM data are small lists here: which record backs which
//! encoder entry (`x86_entries`, `a64_entries`), the public `Feature` names and the LLVM
//! `SubtargetFeature` each one maps to (`x86_features`, `a64_features`; names without an LLVM
//! feature are listed with the reason), and the width/scale of immediate operand classes
//! (`a64_imm_classes`).
const std = @import("std");
const lj = @import("llvm_json.zig");

const A = std.mem.Allocator;

// ------------------------------------------------------------------------------------ features --

/// Public feature name -> LLVM `SubtargetFeature` record, or null with the reason it has none.
const FeatureMap = struct { []const u8, ?[]const u8 };

/// x86 feature names (the spelling used by `Options.target_features` and the encoder tables).
const x86_features = [_]FeatureMap{
    .{ "ADX", "FeatureADX" },
    .{ "AESKLE", null }, // LLVM folds AES Key Locker into FeatureKL
    .{ "AESNI", "FeatureAES" },
    .{ "AMX_AVX512", "FeatureAMXAVX512" },
    .{ "AMX_BF16", "FeatureAMXBF16" },
    .{ "AMX_COMPLEX", "FeatureAMXCOMPLEX" },
    .{ "AMX_FP16", "FeatureAMXFP16" },
    .{ "AMX_FP8", "FeatureAMXFP8" },
    .{ "AMX_INT8", "FeatureAMXINT8" },
    .{ "AMX_MOVRS", "FeatureAMXMOVRS" },
    .{ "AMX_TF32", null }, // removed from LLVM (AMX-TF32 was withdrawn by Intel)
    .{ "AMX_TILE", "FeatureAMXTILE" },
    .{ "AVX", "FeatureAVX" },
    .{ "AVX10_2", "FeatureAVX10_2" },
    .{ "AVX2", "FeatureAVX2" },
    .{ "AVX512_BF16", "FeatureBF16" },
    .{ "AVX512_BITALG", "FeatureBITALG" },
    .{ "AVX512_BW", "FeatureBWI" },
    .{ "AVX512_CD", "FeatureCDI" },
    .{ "AVX512_DQ", "FeatureDQI" },
    .{ "AVX512_F", "FeatureAVX512" },
    .{ "AVX512_FP16", "FeatureFP16" },
    .{ "AVX512_IFMA", "FeatureIFMA" },
    .{ "AVX512_VBMI", "FeatureVBMI" },
    .{ "AVX512_VBMI2", "FeatureVBMI2" },
    .{ "AVX512_VNNI", "FeatureVNNI" },
    .{ "AVX512_VP2INTERSECT", "FeatureVP2INTERSECT" },
    .{ "AVX512_VPOPCNTDQ", "FeatureVPOPCNTDQ" },
    .{ "AVX_IFMA", "FeatureAVXIFMA" },
    .{ "AVX_NE_CONVERT", "FeatureAVXNECONVERT" },
    .{ "AVX_VNNI", "FeatureAVXVNNI" },
    .{ "AVX_VNNI_INT16", "FeatureAVXVNNIINT16" },
    .{ "AVX_VNNI_INT8", "FeatureAVXVNNIINT8" },
    .{ "BMI", "FeatureBMI" },
    .{ "BMI2", "FeatureBMI2" },
    .{ "CET_IBT", null }, // LLVM assembles endbr32/endbr64 unconditionally
    .{ "CET_SS", "FeatureSHSTK" },
    .{ "CLDEMOTE", "FeatureCLDEMOTE" },
    .{ "CLFLUSH", null }, // LLVM assembles clflush unconditionally (CPUID.01H:EDX.CLFSH)
    .{ "CLFLUSHOPT", "FeatureCLFLUSHOPT" },
    .{ "CLWB", "FeatureCLWB" },
    .{ "CLZERO", "FeatureCLZERO" },
    .{ "CMOV", "FeatureCMOV" },
    .{ "CMPCCXADD", "FeatureCMPCCXADD" },
    .{ "CMPXCHG16B", "FeatureCX16" },
    .{ "CMPXCHG8B", "FeatureCX8" },
    .{ "ENQCMD", "FeatureENQCMD" },
    .{ "F16C", "FeatureF16C" },
    .{ "FMA", "FeatureFMA" },
    .{ "FMA4", "FeatureFMA4" },
    .{ "FPU", "FeatureX87" },
    .{ "FSGSBASE", "FeatureFSGSBase" },
    .{ "FXSR", "FeatureFXSR" },
    .{ "GFNI", "FeatureGFNI" },
    .{ "HRESET", "FeatureHRESET" },
    .{ "I486", null }, // baseline for every x86-64 CPU; LLVM has no feature for it
    .{ "INVLPGB", null }, // LLVM assembles invlpgb/tlbsync unconditionally
    .{ "KL", "FeatureKL" },
    .{ "LAHFSAHF", "FeatureLAHFSAHF64" },
    .{ "LWP", "FeatureLWP" },
    .{ "LZCNT", "FeatureLZCNT" },
    .{ "MCOMMIT", null }, // LLVM assembles mcommit unconditionally
    .{ "MMX", "FeatureMMX" },
    .{ "MMX2", null }, // the SSE integer extensions to MMX; LLVM gates them by FeatureSSE1
    .{ "MONITOR", null }, // LLVM assembles monitor/mwait unconditionally (CPUID.01H:ECX.MONITOR)
    .{ "MONITORX", "FeatureMWAITX" },
    .{ "MOVBE", "FeatureMOVBE" },
    .{ "MOVDIR64B", "FeatureMOVDIR64B" },
    .{ "MOVDIRI", "FeatureMOVDIRI" },
    .{ "MOVRS", "FeatureMOVRS" },
    .{ "MSR", null }, // rdmsr/wrmsr are unconditional in LLVM (CPUID.01H:EDX.MSR)
    .{ "MSRLIST", null }, // LLVM assembles rdmsrlist/wrmsrlist unconditionally
    .{ "MSR_IMM", null }, // immediate-form rdmsr/wrmsrns; unconditional in LLVM
    .{ "OSPKE", "FeaturePKU" },
    .{ "PCLMULQDQ", "FeaturePCLMUL" },
    .{ "PCONFIG", "FeaturePCONFIG" },
    .{ "POPCNT", "FeaturePOPCNT" },
    .{ "PREFETCHI", "FeaturePREFETCHI" },
    .{ "PREFETCHW", "FeaturePRFCHW" },
    .{ "PREFETCHWT1", null }, // removed from LLVM
    .{ "PTWRITE", "FeaturePTWRITE" },
    .{ "RAO_INT", "FeatureRAOINT" },
    .{ "RDPID", "FeatureRDPID" },
    .{ "RDPRU", "FeatureRDPRU" },
    .{ "RDRAND", "FeatureRDRAND" },
    .{ "RDSEED", "FeatureRDSEED" },
    .{ "RDTSC", null }, // rdtsc is unconditional in LLVM (CPUID.01H:EDX.TSC)
    .{ "RDTSCP", null }, // rdtscp is unconditional in LLVM (CPUID.80000001H:EDX.RDTSCP)
    .{ "RMPQUERY", null }, // AMD SEV-SNP; LLVM assembles the SNP instructions unconditionally
    .{ "RTM", "FeatureRTM" },
    .{ "SEAM", null }, // Intel TDX host instructions; unconditional in LLVM
    .{ "SERIALIZE", "FeatureSERIALIZE" },
    .{ "SEV_ES", null }, // vmgexit; unconditional in LLVM
    .{ "SEV_SNP", null }, // pvalidate/rmpadjust/rmpupdate/psmash; unconditional in LLVM
    .{ "SHA", "FeatureSHA" },
    .{ "SHA512", "FeatureSHA512" },
    .{ "SKINIT", null }, // AMD SVM skinit/stgi; unconditional in LLVM
    .{ "SM3", "FeatureSM3" },
    .{ "SM4", "FeatureSM4" },
    .{ "SMAP", null }, // clac/stac; unconditional in LLVM
    .{ "SMX", null }, // getsec; unconditional in LLVM
    .{ "SSE", "FeatureSSE1" },
    .{ "SSE2", "FeatureSSE2" },
    .{ "SSE3", "FeatureSSE3" },
    .{ "SSE4A", "FeatureSSE4A" },
    .{ "SSE4_1", "FeatureSSE41" },
    .{ "SSE4_2", "FeatureSSE42" },
    .{ "SSSE3", "FeatureSSSE3" },
    .{ "SVM", null }, // AMD-V vmrun/vmload/...; unconditional in LLVM
    .{ "TBM", "FeatureTBM" },
    .{ "TSE", null }, // pbndkb; unconditional in LLVM
    .{ "TSXLDTRK", "FeatureTSXLDTRK" },
    .{ "UINTR", "FeatureUINTR" },
    .{ "USER_MSR", "FeatureUSERMSR" },
    .{ "VAES", "FeatureVAES" },
    .{ "VMX", null }, // Intel VT-x vmxon/vmcall/...; unconditional in LLVM
    .{ "VPCLMULQDQ", "FeatureVPCLMULQDQ" },
    .{ "WAITPKG", "FeatureWAITPKG" },
    .{ "WBNOINVD", "FeatureWBNOINVD" },
    .{ "WRMSRNS", null }, // unconditional in LLVM
    .{ "XOP", "FeatureXOP" },
    .{ "XSAVE", "FeatureXSAVE" },
    .{ "XSAVEC", "FeatureXSAVEC" },
    .{ "XSAVEOPT", "FeatureXSAVEOPT" },
    .{ "XSAVES", "FeatureXSAVES" },
};

/// AArch64 feature names (Arm `FEAT_*` spelling without the prefix, as in `Options.target_features`).
const a64_features = [_]FeatureMap{
    .{ "AES", "FeatureAES" },
    .{ "ASIMD", "FeatureNEON" },
    .{ "BF16", "FeatureBF16" },
    .{ "BRBE", "FeatureBRBE" },
    .{ "BTI", "FeatureBranchTargetId" },
    .{ "CHK", "FeatureCHK" },
    .{ "CLRBHB", "FeatureCLRBHB" },
    .{ "CPA", "FeatureCPA" },
    .{ "CRC32", "FeatureCRC" },
    .{ "CSSC", "FeatureCSSC" },
    .{ "D128", "FeatureD128" },
    .{ "DGH", null }, // a hint-space instruction; LLVM assembles it unconditionally
    .{ "DOTPROD", "FeatureDotProd" },
    .{ "FAMINMAX", "FeatureFAMINMAX" },
    .{ "FCMA", "FeatureComplxNum" },
    .{ "FHM", "FeatureFP16FML" },
    .{ "FLAGM", "FeatureFlagM" },
    .{ "FLAGM2", "FeatureAltFPCmp" },
    .{ "FP16", "FeatureFullFP16" },
    .{ "FP8", "FeatureFP8" },
    .{ "FP8DOT2", "FeatureFP8DOT2" },
    .{ "FP8DOT4", "FeatureFP8DOT4" },
    .{ "FP8FMA", "FeatureFP8FMA" },
    .{ "FRINTTS", "FeatureFRInt3264" },
    .{ "GCS", "FeatureGCS" },
    .{ "HBC", "FeatureHBC" },
    .{ "I8MM", "FeatureMatMulInt8" },
    .{ "ITE", "FeatureITE" },
    .{ "JSCVT", "FeatureJS" },
    .{ "LOR", "FeatureLOR" },
    .{ "LRCPC", "FeatureRCPC" },
    .{ "LRCPC2", "FeatureRCPC_IMMO" },
    .{ "LRCPC3", "FeatureRCPC3" },
    .{ "LS64", "FeatureLS64" },
    .{ "LS64_ACCDATA", "FeatureLS64" }, // LLVM folds FEAT_LS64_ACCDATA into ls64
    .{ "LS64_V", "FeatureLS64" }, // LLVM folds FEAT_LS64_V into ls64
    .{ "LSE", "FeatureLSE" },
    .{ "LUT", "FeatureLUT" },
    .{ "MOPS", "FeatureMOPS" },
    .{ "MTE", "FeatureMTE" },
    .{ "MTE2", "FeatureMTE" }, // LLVM's mte covers FEAT_MTE2
    .{ "PAUTH", "FeaturePAuth" },
    .{ "RAS", "FeatureRAS" },
    .{ "RDM", "FeatureRDM" },
    .{ "RPRFM", null }, // a hint-space instruction; LLVM assembles it unconditionally
    .{ "SB", "FeatureSB" },
    .{ "SHA1", "FeatureSHA2" }, // LLVM's sha2 covers SHA1 and SHA256
    .{ "SHA256", "FeatureSHA2" },
    .{ "SHA3", "FeatureSHA3" },
    .{ "SHA512", "FeatureSHA3" }, // LLVM's sha3 covers SHA512 and SHA3
    .{ "SM3", "FeatureSM4" }, // LLVM's sm4 covers SM3 and SM4
    .{ "SM4", "FeatureSM4" },
    .{ "SME", "FeatureSME" },
    .{ "SPE", "FeatureSPE" },
    .{ "SPECRES", "FeaturePredRes" },
    .{ "SPECRES2", "FeatureSPECRES2" },
    .{ "SVE", "FeatureSVE" },
    .{ "SVE2", "FeatureSVE2" },
    .{ "SVE2_AES", "FeatureAliasSVE2AES" },
    .{ "SVE2_BITPERM", "FeatureAliasSVE2BitPerm" },
    .{ "SVE2_SHA3", "FeatureAliasSVE2SHA3" },
    .{ "SVE2_SM4", "FeatureAliasSVE2SM4" },
    .{ "SVE_BF16", "FeatureBF16" }, // LLVM gates SVE BF16 forms by sve + bf16
    .{ "SVE_F32MM", "FeatureMatMulFP32" },
    .{ "SVE_F64MM", "FeatureMatMulFP64" },
    .{ "SVE_I8MM", "FeatureMatMulInt8" }, // LLVM gates SVE int8 matmul by sve + i8mm
    .{ "SYSINSTR128", "FeatureD128" }, // LLVM's d128 covers sysp
    .{ "SYSREG128", "FeatureD128" }, // LLVM's d128 covers mrrs/msrr
    .{ "THE", "FeatureTHE" },
    .{ "TME", null }, // removed from LLVM
    .{ "TRF", "FeatureTRACEV8_4" },
    .{ "WFXT", "FeatureWFxT" },
};

fn checkFeatures(db: lj.Db, comptime map: []const FeatureMap, target: []const u8) !void {
    for (map) |m| if (m[1]) |rec| {
        const r = db.get(rec) orelse {
            std.debug.print("{s}: SubtargetFeature {s} (for {s}) not in LLVM\n", .{ target, rec, m[0] });
            return error.MissingRecord;
        };
        if (lj.str(r, "Name").len == 0) return error.NotAFeature;
    };
}

/// Public feature names for the LLVM features in a record's `Predicates`.
fn predFeatures(a: A, db: lj.Db, r: lj.Obj, comptime map: []const FeatureMap) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (lj.list(r, "Predicates")) |p| {
        const pr = db.get(lj.defName(p)) orelse continue;
        const cond = lj.str(pr, "AssemblerCondString");
        // e.g. "X86::FeatureAVX2" or "AArch64::FeatureNEON"; a predicate may name several.
        var it = std.mem.tokenizeAny(u8, cond, " ,()|&!");
        while (it.next()) |tok| {
            const c = std.mem.indexOf(u8, tok, "::") orelse continue;
            const feat = tok[c + 2 ..];
            for (map) |m| if (m[1]) |rec| if (std.mem.eql(u8, rec, feat)) {
                try out.append(a, m[0]);
                break;
            };
        }
    }
    return out.items;
}

fn renderFeatures(a: A, out: *std.ArrayList(u8), comptime map: []const FeatureMap, llvm_td: []const u8) !void {
    try out.print(a,
        \\/// CPU features. Each name maps to an LLVM `SubtargetFeature` of {s} (shown in the comment);
        \\/// names marked "no LLVM feature" are kept for instructions LLVM assembles unconditionally or
        \\/// no longer supports (reasons in tools/gen_isa.zig).
        \\pub const Feature = enum {{
        \\
    , .{llvm_td});
    for (map) |m| {
        const name = if (std.ascii.isDigit(m[0][0])) try std.fmt.allocPrint(a, "@\"{s}\"", .{m[0]}) else m[0];
        if (m[1]) |rec| try out.print(a, "    {s}, // {s}\n", .{ name, rec }) else try out.print(a, "    {s}, // no LLVM feature\n", .{name});
    }
    try out.appendSlice(a,
        \\};
        \\pub const Features = std.EnumSet(Feature);
        \\
        \\/// Set of features from a list (comptime-friendly).
        \\pub fn featureSet(comptime list: []const Feature) Features {
        \\    @setEvalBranchQuota(100_000);
        \\    var s: Features = .empty;
        \\    for (list) |f| s.insert(f);
        \\    return s;
        \\}
        \\
        \\
    );
}

fn featureList(a: A, fs: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (fs, 0..) |f, i| try out.print(a, "{s}.@\"{s}\"", .{ if (i != 0) ", " else "", f });
    return if (fs.len > 1) std.fmt.allocPrint(a, " {s} ", .{out.items}) else out.items;
}

fn zstr(a: A, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(a, "\\\""),
        '\\' => try out.appendSlice(a, "\\\\"),
        '\t' => try out.append(a, ' '),
        else => try out.append(a, c),
    };
    try out.append(a, '"');
    return out.items;
}

fn record(db: lj.Db, name: []const u8) !lj.Obj {
    return db.get(name) orelse {
        std.debug.print("LLVM record {s} not found\n", .{name});
        return error.MissingRecord;
    };
}

// ------------------------------------------------------------------------------------- x86-64 --

const alu_ops = [_][]const u8{ "ADD", "OR", "ADC", "SBB", "AND", "SUB", "XOR", "CMP" };
const cond_codes = [_][]const u8{ "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g" };

/// Encoder entry -> LLVM records, in the order the encoder tries them.
const x86_entries = [_]struct { []const u8, []const []const u8 }{
    .{ "mov_rr", &.{"MOV64rr"} },
    .{ "mov_mr", &.{"MOV64mr"} },
    .{ "mov_rm", &.{"MOV64rm"} },
    .{ "mov_mi", &.{"MOV64mi32"} },
    .{ "lea", &.{"LEA64r"} },
    .{ "mov_ri", &.{ "MOV32ri", "MOV64ri32", "MOV64ri" } },
    .{ "movabs", &.{"MOV64ri"} },
    .{ "test_rr", &.{"TEST64rr"} },
    .{ "imul_rr", &.{"IMUL64rr"} },
    .{ "imul_rm", &.{"IMUL64rm"} },
    .{ "imul_rri", &.{ "IMUL64rri8", "IMUL64rri32" } },
    .{ "push", &.{"PUSH64r"} },
    .{ "pop", &.{"POP64r"} },
    .{ "ret", &.{"RET64"} },
    .{ "nop", &.{"NOOP"} },
    .{ "call_r", &.{"CALL64r"} },
    .{ "jmp_r", &.{"JMP64r"} },
    .{ "jmp", &.{ "JMP_1", "JMP_4" } },
    .{ "call", &.{"CALL64pcrel32"} },
};
/// The eight ALU operations (encoder order = ModRM /digit) as `{op}` + suffix records.
const x86_alu_entries = [_]struct { []const u8, []const []const u8 }{
    .{ "alu_rr", &.{"64rr"} },
    .{ "alu_mr", &.{"64mr"} },
    .{ "alu_rm", &.{"64rm"} },
    .{ "alu_ri", &.{ "64ri8", "64ri32", "64i32" } },
};

fn x86Form(a: A, db: lj.Db, name: []const u8, cc: ?u8) ![]const u8 {
    const r = try record(db, name);
    const form = lj.def(r, "Form");
    var opc: u8 = @intCast(lj.bits(r, "Opcode"));
    var syntax = try lj.asmVariant(a, lj.str(r, "AsmString"), 1);
    if (std.mem.eql(u8, form, "AddCCFrm")) {
        const c = cc orelse return error.MissingCondition;
        opc += c;
        syntax = try std.mem.replaceOwned(u8, a, syntax, "${cond}", cond_codes[c]);
    }
    const map = lj.def(r, "OpMap");
    const opcode = if (std.mem.eql(u8, map, "OB"))
        try std.fmt.allocPrint(a, "0x{X:0>2}", .{opc})
    else if (std.mem.eql(u8, map, "TB"))
        try std.fmt.allocPrint(a, " 0x0F, 0x{X:0>2} ", .{opc})
    else
        return error.UnsupportedOpMap;
    const plus = std.mem.eql(u8, form, "AddRegFrm");
    var digit: ?u8 = null;
    var modrm = false;
    if (std.mem.startsWith(u8, form, "MRMDest") or std.mem.startsWith(u8, form, "MRMSrc")) {
        modrm = true;
    } else if (form.len == 5 and std.mem.startsWith(u8, form, "MRM") and std.ascii.isDigit(form[3]) and (form[4] == 'r' or form[4] == 'm')) {
        modrm = true;
        digit = form[3] - '0';
    } else if (!plus and !std.mem.eql(u8, form, "RawFrm") and !std.mem.eql(u8, form, "AddCCFrm")) {
        std.debug.print("{s}: unsupported Form {s}\n", .{ name, form });
        return error.UnsupportedForm;
    }
    const immt = lj.def(r, "ImmT");
    const imm_map = [_][2][]const u8{ .{ "NoImm", "none" }, .{ "Imm8", "imm8" }, .{ "Imm32", "imm32" }, .{ "Imm32S", "imm32" }, .{ "Imm64", "imm64" }, .{ "Imm8PCRel", "rel8" }, .{ "Imm32PCRel", "rel32" } };
    var imm: ?[]const u8 = null;
    for (imm_map) |m| if (std.mem.eql(u8, m[0], immt)) {
        imm = m[1];
    };
    const imm_kind = imm orelse {
        std.debug.print("{s}: unsupported ImmT {s}\n", .{ name, immt });
        return error.UnsupportedImm;
    };
    const feats = try predFeatures(a, db, r, &x86_features);
    const digit_s = if (digit) |d| try std.fmt.allocPrint(a, "{d}", .{d}) else "null";
    return std.fmt.allocPrint(a, ".{{ .llvm = {s}, .syntax = {s}, .opcode = &.{{{s}}}, .plus = .{s}, .modrm = {}, .digit = {s}, .imm = .{s}, .features = &.{{{s}}} }}", .{
        try zstr(a, name), try zstr(a, syntax), opcode, if (plus) "reg" else "none", modrm, digit_s, imm_kind, try featureList(a, feats),
    });
}

/// `a, b` inside braces as `zig fmt` writes them: `{x}` for one item, `{ a, b }` for more.
fn formList(a: A, items: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (items, 0..) |s, i| {
        if (i != 0) try out.appendSlice(a, ", ");
        try out.appendSlice(a, s);
    }
    return if (items.len > 1) std.fmt.allocPrint(a, " {s} ", .{out.items}) else out.items;
}

pub fn renderX86(a: A, db: lj.Db, header: []const u8) ![]const u8 {
    try checkFeatures(db, &x86_features, "x86");
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, header);
    try out.appendSlice(a,
        \\//! Forms pyrlyn-jit emits for each hand-written x86-64 encoder entry, with the LLVM record
        \\//! that defines the encoding, and the CPU feature set.
        \\const std = @import("std");
        \\
        \\
    );
    try renderFeatures(a, &out, &x86_features, "X86.td");
    try out.appendSlice(a,
        \\pub const Plus = enum { none, reg };
        \\pub const Imm = enum { none, imm8, imm32, imm64, rel8, rel32 };
        \\pub const Form = struct {
        \\    /// LLVM instruction record (llvm/lib/Target/X86/*.td).
        \\    llvm: []const u8,
        \\    /// Intel-syntax assembly string of the record.
        \\    syntax: []const u8,
        \\    opcode: []const u8,
        \\    plus: Plus,
        \\    modrm: bool,
        \\    digit: ?u3,
        \\    imm: Imm,
        \\    features: []const Feature,
        \\};
        \\
        \\pub const table = struct {
        \\
    );
    for (x86_entries) |e| {
        var fs: std.ArrayList([]const u8) = .empty;
        for (e[1]) |n| try fs.append(a, try x86Form(a, db, n, null));
        try out.print(a, "    pub const {s} = [_]Form{{{s}}};\n", .{ e[0], try formList(a, fs.items) });
    }
    for (x86_alu_entries) |e| {
        try out.print(a, "    pub const {s} = [8][]const Form{{\n", .{e[0]});
        for (alu_ops) |op| {
            var fs: std.ArrayList([]const u8) = .empty;
            for (e[1]) |suf| try fs.append(a, try x86Form(a, db, try std.mem.concat(a, u8, &.{ op, suf }), null));
            try out.print(a, "        &.{{{s}}},\n", .{try formList(a, fs.items)});
        }
        try out.appendSlice(a, "    };\n");
    }
    try out.appendSlice(a, "    pub const jcc = [16][]const Form{\n");
    for (0..16) |c| {
        const cc: u8 = @intCast(c);
        try out.print(a, "        &.{{ {s}, {s} }},\n", .{ try x86Form(a, db, "JCC_1", cc), try x86Form(a, db, "JCC_4", cc) });
    }
    try out.appendSlice(a, "    };\n};\n");
    return out.items;
}

// ------------------------------------------------------------------------------------ AArch64 --

/// Encoder entry -> LLVM record, and the `InstAlias` (by Intel/asm string) whose fixed operands
/// apply, if the encoder emits an alias.
const a64_entries = [_]struct { []const u8, []const u8, ?[]const u8 }{
    .{ "movz", "MOVZXi", null },
    .{ "movk", "MOVKXi", null },
    .{ "mov", "ORRXrs", "mov $dst, $src" },
    .{ "add", "ADDXrs", null },
    .{ "sub", "SUBXrs", null },
    .{ "mul", "MADDXrrr", "mul\t$dst, $src1, $src2" },
    .{ "add_imm", "ADDXri", null },
    .{ "sub_imm", "SUBXri", null },
    .{ "cmp", "SUBSXrs", "cmp\t$src1, $src2$sh" },
    .{ "cmp_imm", "SUBSXri", "cmp\t$src, $imm" },
    .{ "ldr", "LDRXui", null },
    .{ "str", "STRXui", null },
    .{ "ret", "RET", null },
    .{ "blr", "BLR", null },
    .{ "b", "B", null },
    .{ "bl", "BL", null },
    .{ "bcond", "Bcc", null },
    .{ "cbz", "CBZX", null },
    .{ "cbnz", "CBNZX", null },
    .{ "sdiv", "SDIVXr", null },
    .{ "udiv", "UDIVXr", null },
    .{ "madd", "MADDXrrr", null },
    .{ "msub", "MSUBXrrr", null },
    .{ "mneg", "MSUBXrrr", "mneg\t$dst, $src1, $src2" },
    .{ "smulh", "SMULHrr", null },
    .{ "umulh", "UMULHrr", null },
    .{ "neg", "SUBXrs", "neg $dst, $src$shift" },
    .{ "and_", "ANDXrs", null },
    .{ "orr", "ORRXrs", null },
    .{ "eor", "EORXrs", null },
    .{ "csel", "CSELXr", null },
    .{ "cset", "CSINCXr", "cset $dst, $cc" },
    .{ "lsl_imm", "UBFMXri", null }, // lsl is UBFM with immr/imms computed from the shift
    .{ "lsr_imm", "UBFMXri", "lsr $dst, $src, $shift" },
    .{ "asr_imm", "SBFMXri", "asr $dst, $src, $shift" },
};

/// Immediate operand classes: field width in bits and scale (bytes per unit). Other classes
/// (shift amounts, bitfield positions, condition codes) are not counted as the immediate.
const a64_imm_classes = [_]struct { []const u8, u8, u8 }{
    .{ "movimm32_imm", 16, 1 },
    .{ "addsub_shifted_imm64", 12, 1 },
    .{ "uimm12s8", 12, 8 },
    .{ "am_b_target", 26, 4 },
    .{ "am_bl_target", 26, 4 },
    .{ "am_brcond", 19, 4 },
};

const Operand = struct { name: []const u8, class: []const u8, fixed: ?u64 };

/// Flattened operand list of a dag (`(outs ...)`, `(ins ...)` or an alias result): complex
/// operands such as `(logical_shifted_reg64 $Rm, $shift)` become their sub-operands.
fn flatten(a: A, v: lj.Value, out: *std.ArrayList(Operand)) !void {
    if (v != .object) return;
    const args = v.object.get("args") orelse return;
    for (args.array.items) |pair_v| {
        const pair = pair_v.array.items;
        const nm: []const u8 = if (pair.len > 1 and pair[1] == .string) pair[1].string else "";
        const x = pair[0];
        switch (x) {
            .integer => |i| try out.append(a, .{ .name = nm, .class = "", .fixed = @intCast(i) }),
            .object => |o| if (o.get("args") != null) {
                try flatten(a, x, out);
            } else {
                const cls = lj.defName(x);
                const fixed: ?u64 = if (std.mem.eql(u8, cls, "XZR") or std.mem.eql(u8, cls, "WZR")) 31 else null;
                try out.append(a, .{ .name = nm, .class = cls, .fixed = fixed });
            },
            else => try out.append(a, .{ .name = nm, .class = "", .fixed = null }),
        }
    }
}

fn findAlias(db: lj.Db, rec: []const u8, syntax: []const u8) !lj.Obj {
    for (db.instancesOf("InstAlias")) |n| {
        const r = db.get(lj.defName(n)) orelse continue;
        if (!std.mem.eql(u8, lj.str(r, "AsmString"), syntax)) continue;
        const res = r.get("ResultInst") orelse continue;
        const op = res.object.get("operator") orelse continue;
        if (std.mem.eql(u8, lj.defName(op), rec)) return r;
    }
    std.debug.print("InstAlias \"{s}\" -> {s} not found\n", .{ syntax, rec });
    return error.MissingRecord;
}

fn a64Form(a: A, db: lj.Db, rec: []const u8, alias: ?[]const u8) ![]const u8 {
    const r = try record(db, rec);
    var base: std.ArrayList(Operand) = .empty;
    try flatten(a, r.get("OutOperandList").?, &base);
    try flatten(a, r.get("InOperandList").?, &base);
    // Inherited tied sources (movk: $Rd = $src) are not encoding fields; drop duplicates by name.
    const ops = base.items;
    var syntax = lj.str(r, "AsmString");
    var feat_rec = r;
    if (alias) |s| {
        const al = try findAlias(db, rec, s);
        var res: std.ArrayList(Operand) = .empty;
        try flatten(a, al.get("ResultInst").?, &res);
        if (res.items.len != ops.len) return error.AliasArity;
        for (ops, res.items) |*b, x| {
            if (x.fixed) |f| b.fixed = f;
            if (x.class.len != 0 and b.class.len == 0) b.class = x.class;
        }
        syntax = s;
        feat_rec = al;
    }
    // Fixed bits from Inst, plus the alias's fixed operands.
    const inst = lj.list(r, "Inst");
    if (inst.len != 32) return error.NotA32BitEncoding;
    var mask: u32 = 0;
    var value: u32 = 0;
    for (inst, 0..) |b, i| {
        const bit: u32 = @as(u32, 1) << @intCast(i);
        switch (b) {
            .integer => |v| {
                mask |= bit;
                if (v == 1) value |= bit;
            },
            .object => |o| {
                const vname = lj.str(o, "var");
                const idx = lj.int(o, "index");
                for (ops) |op| if (op.fixed != null and std.mem.eql(u8, op.name, vname)) {
                    mask |= bit;
                    if ((op.fixed.? >> @intCast(idx)) & 1 != 0) value |= bit;
                };
            },
            else => {},
        }
    }
    var rd_sp = false;
    var rn_sp = false;
    var imm_bits: u8 = 0;
    var imm_scale: u8 = 1;
    for (ops) |op| {
        const sp = std.mem.eql(u8, op.class, "GPR64sp");
        if (op.fixed == null and sp and (std.mem.eql(u8, op.name, "Rd") or std.mem.eql(u8, op.name, "Rt"))) rd_sp = true;
        if (op.fixed == null and sp and std.mem.eql(u8, op.name, "Rn")) rn_sp = true;
        for (a64_imm_classes) |c| if (std.mem.eql(u8, c[0], op.class)) {
            imm_bits = c[1];
            imm_scale = c[2];
        };
    }
    var feats = try predFeatures(a, db, r, &a64_features);
    if (alias != null) feats = try std.mem.concat(a, []const u8, &.{ feats, try predFeatures(a, db, feat_rec, &a64_features) });
    return std.fmt.allocPrint(a, ".{{ .llvm = {s}, .syntax = {s}, .mask = 0x{X:0>8}, .value = 0x{X:0>8}, .rd_sp = {}, .rn_sp = {}, .imm_bits = {d}, .imm_scale = {d}, .features = &.{{{s}}} }}", .{
        try zstr(a, rec), try zstr(a, syntax), mask, value, rd_sp, rn_sp, imm_bits, imm_scale, try featureList(a, feats),
    });
}

pub fn renderA64(a: A, db: lj.Db, header: []const u8) ![]const u8 {
    try checkFeatures(db, &a64_features, "aarch64");
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, header);
    try out.appendSlice(a,
        \\//! Encoding of each hand-written AArch64 encoder entry, from the LLVM record (and alias) that
        \\//! defines it, and the CPU feature set.
        \\const std = @import("std");
        \\
        \\
    );
    try renderFeatures(a, &out, &a64_features, "AArch64.td");
    try out.appendSlice(a,
        \\pub const Form = struct {
        \\    /// LLVM instruction record (llvm/lib/Target/AArch64/*.td).
        \\    llvm: []const u8,
        \\    /// Assembly string of the record, or of the `InstAlias` the encoder emits.
        \\    syntax: []const u8,
        \\    /// Fixed bits of the encoding: `(word & mask) == value` for a correct encoding.
        \\    mask: u32,
        \\    value: u32,
        \\    /// Register 31 in Rd / Rn means SP (otherwise XZR).
        \\    rd_sp: bool,
        \\    rn_sp: bool,
        \\    /// Width of the immediate/offset/displacement field and its scale (bytes per unit).
        \\    imm_bits: u8,
        \\    imm_scale: u8,
        \\    features: []const Feature,
        \\};
        \\
        \\pub const table = struct {
        \\
    );
    for (a64_entries) |e| try out.print(a, "    pub const {s}: Form = {s};\n", .{ e[0], try a64Form(a, db, e[1], e[2]) });
    try out.appendSlice(a, "};\n");
    return out.items;
}
