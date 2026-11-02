#!/usr/bin/env python3
"""Writes src/riscv_tests.zig: every RISC-V emitter (RV64/RV32 I, M, A, F, D, C, pseudo
instructions) checked byte-for-byte against LLVM's assembler (`llvm-mc -show-encoding`).

LLVM comes from mise (`conda:llvm-tools`); override with LLVM_MC=/path/to/llvm-mc.
32-bit forms are assembled without +c (llvm-mc would compress them); `c.*` forms with +c."""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.environ.get("LLVM_MC")
MC_CMD = [MC] if MC else ["mise", "exec", "conda:llvm-tools@23.1.4", "--", "llvm-mc"]
ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
       "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
FABI = ["ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
        "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"]
X3 = [(10, 11, 12), (0, 31, 1), (31, 5, 30), (8, 15, 9), (1, 2, 3)]
IMM12 = [0, 1, -1, 2047, -2048, 100, -100]

# (xlen, compressed, zig statement, asm text)
cases = []
def c(z, asm, xlen=64, comp=False):
    cases.append((xlen, comp, z, asm))

def x(i): return f".{ABI[i]}" if ABI[i] not in ("and", "or") else f'.@"{ABI[i]}"'
def f(i): return f".f{i}"

for xl in (64, 32):
    R = [("add", "add"), ("sub", "sub"), ("sll", "sll"), ("slt", "slt"), ("sltu", "sltu"), ("xor", "xor"),
         ("srl", "srl"), ("sra", "sra"), ('@"or"', "or"), ('@"and"', "and"), ("mul", "mul"), ("mulh", "mulh"),
         ("mulhu", "mulhu"), ("div", "div"), ("divu", "divu"), ("rem", "rem"), ("remu", "remu")]
    if xl == 64:
        R += [("addw", "addw"), ("subw", "subw"), ("mulw", "mulw")]
    for zn, an in R:
        for d, s1, s2 in X3:
            c(f"try a.{zn}({x(d)}, {x(s1)}, {x(s2)});", f"{an} {ABI[d]}, {ABI[s1]}, {ABI[s2]}", xl)
    I = ["addi", "slti", "sltiu", "xori", "ori", "andi"] + (["addiw"] if xl == 64 else [])
    for n in I:
        for (d, s1, _), im in zip(X3, IMM12):
            c(f"try a.{n}({x(d)}, {x(s1)}, {im});", f"{n} {ABI[d]}, {ABI[s1]}, {im}", xl)
    mx = 63 if xl == 64 else 31
    for n in ("slli", "srli", "srai"):
        for (d, s1, _), sh in zip(X3, (0, 1, mx, 31, 17)):
            c(f"try a.{n}({x(d)}, {x(s1)}, {sh});", f"{n} {ABI[d]}, {ABI[s1]}, {sh}", xl)
    for n in ("lui", "auipc"):
        for (d, _, _), im in zip(X3, (0, 1, 0xFFFFF, 0x12345, 0x80000)):
            c(f"try a.{n}({x(d)}, 0x{im:x});", f"{n} {ABI[d]}, 0x{im:x}", xl)
    L = ["lb", "lh", "lw", "lbu", "lhu"] + (["ld", "lwu"] if xl == 64 else [])
    S = ["sb", "sh", "sw"] + (["sd"] if xl == 64 else [])
    for n in L + S + ["jalr"]:
        for (d, s1, _), im in zip(X3, IMM12):
            c(f"try a.{n}({x(d)}, {im}, {x(s1)});", f"{n} {ABI[d]}, {im}({ABI[s1]})", xl)
    # A extension
    amos = ["add", "swap", "xor", "or", "and", "min", "max", "minu", "maxu"]
    for w in (["w", "d"] if xl == 64 else ["w"]):
        for i, op in enumerate(amos):
            ordv, ords = [("none", ""), ("aq", ".aq"), ("rl", ".rl"), ("aqrl", ".aqrl")][i % 4]
            d, s2, s1 = X3[i % 5]
            zop = f'.@"{op}"' if op in ("or", "and") else f".{op}"
            c(f"try a.amo({zop}, .{w}, .{ordv}, {x(d)}, {x(s2)}, {x(s1)});", f"amo{op}.{w}{ords} {ABI[d]}, {ABI[s2]}, ({ABI[s1]})", xl)
        c(f"try a.lr(.{w}, .aq, .a0, .a1);", f"lr.{w}.aq a0, (a1)", xl)
        c(f"try a.sc(.{w}, .rl, .a2, .a3, .a1);", f"sc.{w}.rl a2, a3, (a1)", xl)
    for (p, s_), txt in (((3, 3), "rw, rw"), ((15, 15), "iorw, iorw"), ((1, 2), "w, r"), ((8, 4), "i, o")):
        c(f"try a.fence({p}, {s_});", f"fence {txt}", xl)
    # F/D
    for fmt in ("s", "d"):
        for (d, s1, s2) in X3[:3]:
            for op in ("add", "sub", "mul", "div"):
                for rm in ("dyn", "rne", "rtz"):
                    sfx = "" if rm == "dyn" else f", {rm}"
                    c(f"try a.farith(.{op}, .{fmt}, {f(d)}, {f(s1)}, {f(s2)}, .{rm});", f"f{op}.{fmt} {FABI[d]}, {FABI[s1]}, {FABI[s2]}{sfx}", xl)
            c(f"try a.fsqrt(.{fmt}, {f(d)}, {f(s1)}, .rup);", f"fsqrt.{fmt} {FABI[d]}, {FABI[s1]}, rup", xl)
            c(f"try a.fmin(.{fmt}, {f(d)}, {f(s1)}, {f(s2)});", f"fmin.{fmt} {FABI[d]}, {FABI[s1]}, {FABI[s2]}", xl)
            c(f"try a.fmax(.{fmt}, {f(d)}, {f(s1)}, {f(s2)});", f"fmax.{fmt} {FABI[d]}, {FABI[s1]}, {FABI[s2]}", xl)
            for k, nm in enumerate(("fsgnj", "fsgnjn", "fsgnjx")):
                c(f"try a.fsgnj(.{fmt}, {k}, {f(d)}, {f(s1)}, {f(s2)});", f"{nm}.{fmt} {FABI[d]}, {FABI[s1]}, {FABI[s2]}", xl)
            c(f"try a.fmadd(.{fmt}, {f(d)}, {f(s1)}, {f(s2)}, {f(31)}, .rmm);", f"fmadd.{fmt} {FABI[d]}, {FABI[s1]}, {FABI[s2]}, {FABI[31]}, rmm", xl)
            for cc in ("eq", "lt", "le"):
                c(f"try a.fcmp(.{cc}, .{fmt}, {x(d)}, {f(s1)}, {f(s2)});", f"f{cc}.{fmt} {ABI[d]}, {FABI[s1]}, {FABI[s2]}", xl)
            for k in (["w", "wu", "l", "lu"] if xl == 64 else ["w", "wu"]):
                c(f"try a.fcvtToInt(.{k}, .{fmt}, {x(d)}, {f(s1)}, .rtz);", f"fcvt.{k}.{fmt} {ABI[d]}, {FABI[s1]}, rtz", xl)
                c(f"try a.fcvtFromInt(.{k}, .{fmt}, {f(d)}, {x(s1)}, .rdn);", f"fcvt.{fmt}.{k} {FABI[d]}, {ABI[s1]}, rdn", xl)
            for im in (0, -4, 2047, -2048):
                c(f"try a.fload(.{fmt}, {f(d)}, {im}, {x(s1)});", f"fl{'w' if fmt == 's' else 'd'} {FABI[d]}, {im}({ABI[s1]})", xl)
                c(f"try a.fstore(.{fmt}, {f(d)}, {im}, {x(s1)});", f"fs{'w' if fmt == 's' else 'd'} {FABI[d]}, {im}({ABI[s1]})", xl)
        if fmt == "s" or xl == 64:
            nm = "w" if fmt == "s" else "d"
            c(f"try a.fmvToInt(.{fmt}, .a0, .f10);", f"fmv.x.{nm} a0, fa0", xl)
            c(f"try a.fmvFromInt(.{fmt}, .f10, .a0);", f"fmv.{nm}.x fa0, a0", xl)
    c("try a.fcvtFmt(.d, .f4, .f5, .dyn);", "fcvt.d.s ft4, ft5, dyn", xl)
    # LLVM's default rounding mode for the exact widening conversion is rne, not dyn.
    c("try a.fcvtFmt(.d, .f4, .f5, .rne);", "fcvt.d.s ft4, ft5", xl)
    c("try a.fcvtFmt(.s, .f4, .f5, .rtz);", "fcvt.s.d ft4, ft5, rtz", xl)
    # Branches / jal (resolved backward to a label `back` bytes behind).
    for cond in ("eq", "ne", "lt", "ge", "ltu", "geu"):
        c(f"try branchBack(a, .{cond}, .a0, .a5, 0);", f"b{cond} a0, a5, 0", xl)
        c(f"try branchBack(a, .{cond}, .t6, .zero, 2048);", f"b{cond} t6, zero, -2048", xl)
    c("try jalBack(a, .ra, 0);", "jal ra, 0", xl)
    c("try jalBack(a, .zero, 4096);", "jal zero, -4096", xl)
    # Pseudo-instructions.
    c("try a.nop();", "nop", xl)
    c("try a.mv(.s0, .sp);", "mv s0, sp", xl)
    c("try a.ret();", "ret", xl)
    # Compressed (explicit).
    C = [("try a.cNop();", "c.nop"), ("try a.cAddi(.a0, 1);", "c.addi a0, 1"), ("try a.cAddi(.sp, -32);", "c.addi sp, -32"),
         ("try a.cLi(.t6, -32);", "c.li t6, -32"), ("try a.cLi(.a0, 31);", "c.li a0, 31"), ("try a.cLui(.a0, 1);", "c.lui a0, 1"),
         ("try a.cLui(.t6, -1);", "c.lui t6, 0xfffff"), ("try a.cMv(.a0, .t6);", "c.mv a0, t6"), ("try a.cAdd(.ra, .s11);", "c.add ra, s11"),
         ("try a.cJr(.t0);", "c.jr t0"), ("try a.cJalr(.a0);", "c.jalr a0"), ("try a.cRet();", "c.jr ra"), ("try a.cEbreak();", "c.ebreak"),
         ("try a.cLwsp(.ra, 252);", "c.lwsp ra, 252(sp)"), ("try a.cSwsp(.t6, 252);", "c.swsp t6, 252(sp)"), ("try a.cLwsp(.a0, 0);", "c.lwsp a0, 0(sp)"),
         ("try a.cAddi16sp(-512);", "c.addi16sp sp, -512"), ("try a.cAddi16sp(496);", "c.addi16sp sp, 496"), ("try a.cAddi16sp(16);", "c.addi16sp sp, 16"),
         ("try a.cAddi4spn(.s0, 4);", "c.addi4spn s0, sp, 4"), ("try a.cAddi4spn(.a5, 1020);", "c.addi4spn a5, sp, 1020"),
         ("try a.cLw(.a0, 124, .s1);", "c.lw a0, 124(s1)"), ("try a.cSw(.s0, 4, .a5);", "c.sw s0, 4(a5)"),
         ("try a.cSrli(.a0, 1);", "c.srli a0, 1"), ("try a.cSrai(.s1, 31);", "c.srai s1, 31"), ("try a.cAndi(.a3, -32);", "c.andi a3, -32"),
         ("try a.cSub(.a0, .a1);", "c.sub a0, a1"), ("try a.cXor(.s0, .a5);", "c.xor s0, a5"), ("try a.cOr(.a5, .s0);", "c.or a5, s0"), ("try a.cAnd(.s1, .s1);", "c.and s1, s1"),
         ("try a.cSlli(.t6, 31);", "c.slli t6, 31"), ("try a.cSlli(.a0, 1);", "c.slli a0, 1")]
    for z, asm in C:
        c(z, asm, xl, True)
    for z, asm in [("try cBranchBack(a, true, .a0, 0);", "c.beqz a0, 0"), ("try cBranchBack(a, false, .s1, 256);", "c.bnez s1, -256"),
                   ("try cJBack(a, 0);", "c.j 0"), ("try cJBack(a, 2048);", "c.j -2048")]:
        c(z, asm, xl, True)
    for z, asm in [("try a.cFld(.f8, 248, .a5);", "c.fld fs0, 248(a5)"), ("try a.cFsd(.f15, 0, .s0);", "c.fsd fa5, 0(s0)"),
                   ("try a.cFldsp(.f31, 504);", "c.fldsp ft11, 504(sp)"), ("try a.cFsdsp(.f0, 8);", "c.fsdsp ft0, 8(sp)")]:
        c(z, asm, xl, True)
    if xl == 32:
        for z, asm in [("try a.cFlw(.f8, 124, .a5);", "c.flw fs0, 124(a5)"), ("try a.cFsw(.f15, 4, .s0);", "c.fsw fa5, 4(s0)"),
                       ("try a.cFlwsp(.f31, 252);", "c.flwsp ft11, 252(sp)"), ("try a.cFswsp(.f1, 4);", "c.fswsp ft1, 4(sp)")]:
            c(z, asm, xl, True)
    if xl == 64:
        for z, asm in [("try a.cAddiw(.a0, -1);", "c.addiw a0, -1"), ("try a.cAddiw(.t6, 31);", "c.addiw t6, 31"),
                       ("try a.cLdsp(.ra, 504);", "c.ldsp ra, 504(sp)"), ("try a.cSdsp(.t6, 504);", "c.sdsp t6, 504(sp)"),
                       ("try a.cLd(.a0, 248, .s1);", "c.ld a0, 248(s1)"), ("try a.cSd(.s0, 8, .a5);", "c.sd s0, 8(a5)"),
                       ("try a.cSubw(.a0, .a1);", "c.subw a0, a1"), ("try a.cAddw(.a5, .s0);", "c.addw a5, s0"),
                       ("try a.cSlli(.t6, 63);", "c.slli t6, 63"), ("try a.cSrli(.a0, 63);", "c.srli a0, 63")]:
            c(z, asm, xl, True)
    else:
        c("try cJalBack(a, 0);", "c.jal 0", xl, True)

def mc(lines, xlen, comp):
    attrs = "+m,+a,+f,+d" + (",+c" if comp else "")
    r = subprocess.run(MC_CMD + [f"-triple=riscv{xlen}", f"-mattr={attrs}", "-show-encoding"],
                       input="\n".join(lines) + "\n", capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    enc = re.findall(r"encoding: \[([^\]]*)\]", r.stdout)
    assert len(enc) == len(lines), (len(enc), len(lines), r.stdout[-500:])
    return [[int(b, 16) for b in e.split(",")] for e in enc]

# Auto-compress: the 32-bit forms through `AssemblerT(.{ .compress = true })` vs llvm-mc with +c
# (LLVM compresses whenever an equivalent 16-bit form exists). Label branches are excluded:
# the assembler never compresses them.
for xl, comp, z, asm in list(cases):
    if not comp and "Back(" not in z:
        cases.append((xl, "auto", z, asm))

groups = {}
for i, (xl, comp, z, asm) in enumerate(cases):
    groups.setdefault((xl, comp), []).append(i)
enc = {}
for (xl, comp), idx in groups.items():
    for i, b in zip(idx, mc([cases[i][3] for i in idx], xl, comp)):
        enc[i] = b

out = ["//! GENERATED by scripts/gen_riscv_tests.py — do not edit. Expected bytes from LLVM",
       "//! (`llvm-mc -show-encoding`, -mattr=+m,+a,+f,+d[,+c]) for every RISC-V emitter.",
       'const std = @import("std");', 'const rv = @import("riscv/assembler.zig");',
       'const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;', "",
       "/// Binds a label, then pads `back` bytes so a following branch targets `-back`.",
       "fn label(a: anytype, back: u32) !rv.Label {",
       "    const l = try a.newLabel();", "    try a.bind(l);",
       "    var i: u32 = 0;", "    while (i < back) : (i += 2) try a.buf.emitBytes(&.{ 0x01, 0x00 });", "    return l;", "}",
       "fn branchBack(a: anytype, c: rv.Cond, r1: rv.X, r2: rv.X, back: u32) !void {",
       "    try a.b(c, r1, r2, try label(a, back));", "}",
       "fn jalBack(a: anytype, rd: rv.X, back: u32) !void {", "    try a.jal(rd, try label(a, back));", "}",
       "fn cBranchBack(a: anytype, z: bool, r: rv.X, back: u32) !void {",
       "    const l = try label(a, back);", "    if (z) try a.cBeqz(r, l) else try a.cBnez(r, l);", "}",
       "fn cJBack(a: anytype, back: u32) !void {", "    try a.cJ(try label(a, back));", "}",
       "fn cJalBack(a: anytype, back: u32) !void {", "    try a.cJal(try label(a, back));", "}", "",
       "fn tail(cb: *CodeBuffer, expected: []const u8, text: []const u8) u32 {",
       "    defer cb.reset();", "    const got = cb.code()[cb.code().len -| expected.len ..];",
       "    if (std.mem.eql(u8, expected, got)) return 0;",
       '    std.debug.print("riscv mismatch: {s}: expected {x}, got {x}\\n", .{ text, expected, got });',
       "    return 1;", "}", ""]
for (xl, comp), idx in groups.items():
    asm_t = ("rv.Assembler" if xl == 64 else "rv.Assembler32") if comp != "auto" else f"rv.AssemblerT(.{{ .xlen = .rv{xl}, .compress = true }})"
    label = {False: "", True: "C", "auto": " auto-compress"}[comp]
    out += [f'test "riscv vs llvm-mc: RV{xl}{label} ({len(idx)} cases)" {{',
            "    @setEvalBranchQuota(100_000);",
            "    var cb = CodeBuffer.init(std.testing.allocator);", "    defer cb.deinit();",
            f"    const a = {asm_t}.init(&cb);", "    var fails: u32 = 0;"]
    for i in idx:
        _, _, z, asm = cases[i]
        bs = ", ".join(f"0x{b:02x}" for b in enc[i])
        out += [f"    {z}", f'    fails += tail(&cb, &.{{ {bs} }}, "{asm}");']
    out += ["    try std.testing.expectEqual(@as(u32, 0), fails);", "}", ""]
open(os.path.join(ROOT, "src/riscv_tests.zig"), "w").write("\n".join(out))
print(f"{len(cases)} cases")
