#!/usr/bin/env python3
"""Generate src/enc_tests.zig: byte-exact encoding tests checked against the system assembler.

Each case pairs a pyrlyn-jit call with the equivalent assembly text. The script assembles all
cases with `clang -c` (LLVM integrated assembler) and reads the bytes back with `objdump -d`,
so the expected bytes come from LLVM, not from pyrlyn-jit. Re-run after adding forms:

    python3 scripts/gen_enc_tests.py && zig build test
"""
import os, re, subprocess, tempfile

GP = "rax rcx rdx rbx rsp rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15".split()
D32 = {"rax": "eax", "rcx": "ecx", "rdx": "edx", "rbx": "ebx", "rsp": "esp", "rbp": "ebp", "rsi": "esi", "rdi": "edi"}
ALU = "add or adc sbb and sub xor cmp".split()
SC = {"x1": 1, "x2": 2, "x4": 4, "x8": 8}


def zop(op):
    return f'@"{op}"' if op in ("or", "and") else op


def r32(r):
    return D32.get(r, r + "d")


def zmem(base, idx=None, sc="x1", disp=0):
    if base is None:
        return f"Mem{{ .index = .{idx}, .scale = .{sc}, .disp = {disp} }}"
    if idx is None:
        return f"ptr(.{base}, {disp})"
    return f"ptrIdx(.{base}, .{idx}, .{sc}, {disp})"


def amem(base, idx=None, sc="x1", disp=0):
    parts = []
    if base:
        parts.append(base)
    if idx:
        parts.append(f"{idx}*{SC[sc]}")
    s = "+".join(parts) if parts else ""
    if base is None and idx is None:
        s = str(disp)
    elif disp:
        s += f"{disp:+d}"
    return f"qword ptr [{s}]"


def x86_cases():
    c = []
    for d in GP:
        for s in ("rax", "rsp", "r8", "r15"):
            c.append((f"movRR, .{{ Gp.{d}, Gp.{s} }}", f"mov {d}, {s}"))
    for r in GP:
        c += [(f"push, .{{Gp.{r}}}", f"push {r}"), (f"pop, .{{Gp.{r}}}", f"pop {r}"),
              (f"callR, .{{Gp.{r}}}", f"call {r}"), (f"jmpR, .{{Gp.{r}}}", f"jmp {r}")]
        for v in (0, 1, 127, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            c.append((f"movRI, .{{ Gp.{r}, @as(i64, {v}) }}", f"mov {r32(r)}, {v}"))
        for v in (-1, -128, -0x80000000):
            c.append((f"movRI, .{{ Gp.{r}, @as(i64, {v}) }}", f"mov {r}, {v}"))
        for v in (0x100000000, -0x80000001, -0x8000000000000000):
            c.append((f"movRI, .{{ Gp.{r}, @as(i64, {v}) }}", f"movabs {r}, {v}"))
        c.append((f"movabs, .{{ Gp.{r}, @as(u64, 7) }}", f"movabs {r}, 7"))
    for op in ALU:
        for d, s in (("rax", "rcx"), ("r9", "rdi"), ("rsp", "r15")):
            c.append((f"aluRR, .{{ Alu.{zop(op)}, Gp.{d}, Gp.{s} }}", f"{op} {d}, {s}"))
        for d in ("rax", "rcx", "r10", "rsp"):
            for v in (0, 1, -1, 127, -128, 128, -129, 0x7FFFFFFF, -0x80000000):
                c.append((f"aluRI, .{{ Alu.{zop(op)}, Gp.{d}, @as(i32, {v}) }}", f"{op} {d}, {v}"))
        c.append((f"aluRM, .{{ Alu.{zop(op)}, Gp.r11, ptr(.rbp, -8) }}", f"{op} r11, {amem('rbp', disp=-8)}"))
        c.append((f"aluMR, .{{ Alu.{zop(op)}, ptrIdx(.rsp, .r12, .x8, 64), Gp.rdx }}", f"{op} {amem('rsp', 'r12', 'x8', 64)}, rdx"))
    for d, s in (("rax", "rcx"), ("r12", "r13"), ("rsp", "rbp")):
        c += [(f"testRR, .{{ Gp.{d}, Gp.{s} }}", f"test {d}, {s}"),
              (f"imulRR, .{{ Gp.{d}, Gp.{s} }}", f"imul {d}, {s}")]
        for v in (3, -128, 127, 128, -129, 0x7FFFFFFF):
            c.append((f"imulRRI, .{{ Gp.{d}, Gp.{s}, @as(i32, {v}) }}", f"imul {d}, {s}, {v}"))
    mems = []
    for b in GP:
        for disp in (0, 8, -128, 127, 128, -129, 0x7FFFFFFF, -0x80000000):
            mems.append((b, None, "x1", disp))
    for b in ("rax", "rsp", "rbp", "r12", "r13", "r15"):
        for i in GP:
            if i == "rsp":
                continue
            for sc in SC:
                mems.append((b, i, sc, 0 if sc != "x8" else 300))
    for i in ("rax", "rbp", "r12", "r13"):
        mems.append((None, i, "x4", 16))
    for m in mems:
        zm, am = zmem(*m), amem(*m)
        c += [(f"movRM, .{{ Gp.rax, {zm} }}", f"mov rax, {am}"),
              (f"movRM, .{{ Gp.r9, {zm} }}", f"mov r9, {am}")]
    for m in mems[::7]:
        zm, am = zmem(*m), amem(*m)
        c += [(f"movMR, .{{ {zm}, Gp.r14 }}", f"mov {am}, r14"),
              (f"movMI, .{{ {zm}, @as(i32, -2) }}", f"mov {am}, -2"),
              (f"lea, .{{ Gp.r8, {zm} }}", "lea r8, " + am.replace("qword ptr ", "")),
              (f"imulRM, .{{ Gp.rbx, {zm} }}", f"imul rbx, {am}")]
    c += [("ret, .{}", "ret"), ("nop, .{}", "nop")]
    return c


def a64_cases():
    c = []
    X = [f"x{i}" for i in (0, 1, 7, 15, 16, 28, 29, 30)]
    for r in X:
        for sh in (0, 16, 32, 48):
            c.append((f"movz, .{{ X.{r}, @as(u16, 0xBEEF), @as(u6, {sh}) }}", f"movz {r}, #0xbeef, lsl #{sh}"))
            c.append((f"movk, .{{ X.{r}, @as(u16, 1), @as(u6, {sh}) }}", f"movk {r}, #1, lsl #{sh}"))
        c.append((f"mov, .{{ X.{r}, X.x3 }}", f"mov {r}, x3"))
        c.append((f"mov, .{{ X.x3, X.{r} }}", f"mov x3, {r}"))
        for op in ("add", "sub", "mul"):
            c.append((f"{op}, .{{ X.{r}, X.x30, X.x1 }}", f"{op} {r}, x30, x1"))
        for op in ("addImm", "subImm"):
            for v in (0, 1, 4095):
                c.append((f"{op}, .{{ X.{r}, X.x2, @as(u12, {v}) }}", f"{op[:3]} {r}, x2, #{v}"))
        c.append((f"cmp, .{{ X.{r}, X.x9 }}", f"cmp {r}, x9"))
        c.append((f"cmpImm, .{{ X.{r}, @as(u12, 4095) }}", f"cmp {r}, #4095"))
        for off in (0, 8, 32760):
            c.append((f"ldr, .{{ X.{r}, X.x4, @as(u15, {off}) }}", f"ldr {r}, [x4, #{off}]"))
            c.append((f"str, .{{ X.{r}, X.x4, @as(u15, {off}) }}", f"str {r}, [x4, #{off}]"))
        c.append((f"blr, .{{X.{r}}}", f"blr {r}"))
    c += [("addImm, .{ X.zr_sp, X.zr_sp, @as(u12, 16) }", "add sp, sp, #16"),
          ("subImm, .{ X.zr_sp, X.zr_sp, @as(u12, 32) }", "sub sp, sp, #32"),
          ("cmpImm, .{ X.zr_sp, @as(u12, 1) }", "cmp sp, #1"),
          ("ldr, .{ X.x0, X.zr_sp, @as(u15, 16) }", "ldr x0, [sp, #16]"),
          ("str, .{ X.x30, X.zr_sp, @as(u15, 8) }", "str x30, [sp, #8]"),
          ("mov, .{ X.x0, X.zr_sp }", "mov x0, xzr"),
          ("ret, .{}", "ret")]
    return c


def assemble(target, lines, intel):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.s")
        with open(src, "w") as f:
            if intel:
                f.write(".intel_syntax noprefix\n")
            f.write("\n".join(lines) + "\n")
        obj = os.path.join(d, "t.o")
        subprocess.run(["clang", "-c", "-target", target, src, "-o", obj], check=True)
        out = subprocess.run(["objdump", "-d", obj], check=True, capture_output=True, text=True).stdout
    res = []
    for ln in out.splitlines():
        m = re.match(r"\s*[0-9a-f]+:((?:\s[0-9a-f]{2}(?=\s))+)", ln)
        w = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", ln)
        if w and not intel:  # AArch64: one 32-bit word, printed big-endian
            res.append(list(int(w.group(1), 16).to_bytes(4, "little")))
        elif m:
            res.append([int(b, 16) for b in m.group(1).split()])
    assert len(res) == len(lines), (len(res), len(lines))
    return res


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = ["//! GENERATED by scripts/gen_enc_tests.py — do not edit. Expected bytes come from the",
           "//! LLVM assembler (clang -c + objdump), one case per instruction form and operand class.",
           'const std = @import("std");', 'const opts = @import("pyrlyn_jit_options");',
           'const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;', "",
           "/// Runs one case twice: plain assembler, and with validation (operand checks + the emitted",
           "/// bytes checked against the LLVM-derived forms in isa.zig).",
           "fn check(comptime A: type, expected: []const u8, comptime name: []const u8, args: anytype, asm_text: []const u8) !void {",
           "    inline for (.{ A.Plain, A.Checked }) |T| {",
           "        var cb = CodeBuffer.init(std.testing.allocator);", "        defer cb.deinit();",
           "        @call(.auto, @field(T, name), .{T.init(&cb)} ++ args) catch |e| {",
           '            std.debug.print("{s}: {s}\\n", .{ @errorName(e), asm_text });', "            return e;", "        };",
           "        std.testing.expectEqualSlices(u8, expected, cb.code()) catch |e| {",
           '            std.debug.print("encoding mismatch: {s}\\n", .{asm_text});', "            return e;", "        };",
           "    }", "}", ""]
    for name, target, cases, intel, mod, decls in (
            ("x86_64", "x86_64-apple-macos", x86_cases(), True, "x86_64/assembler.zig",
             ["Gp", "Alu", "Mem", "ptr", "ptrIdx"]),
            ("aarch64", "arm64-apple-macos", a64_cases(), False, "aarch64/assembler.zig", ["X"])):
        bs = assemble(target, [a for _, a in cases], intel)
        out.append(f"test \"{name}: byte-exact vs LLVM ({len(cases)} cases)\" {{")
        out.append(f"    if (!opts.{name}) return error.SkipZigTest;")
        out.append("    @setEvalBranchQuota(100_000);")
        out.append(f'    const m = @import("{mod}");')
        for d in decls:
            out.append(f"    const {d} = m.{d};")
        out.append("    const A = struct {")
        out.append("        const Plain = m.AssemblerT(.{});")
        out.append("        const Checked = m.AssemblerT(.{ .validate = true });")
        out.append("    };")
        for (call, a), b in zip(cases, bs):
            fn, args = call.split(", ", 1)
            hexs = ", ".join(f"0x{x:02X}" for x in b)
            out.append(f'    try check(A, &.{{ {hexs} }}, "{fn}", {args}, "{a}");')
        out.append("}")
        out.append("")
    with open(os.path.join(root, "src", "enc_tests.zig"), "w") as f:
        f.write("\n".join(out))
    subprocess.run(["zig", "fmt", os.path.join(root, "src", "enc_tests.zig")], check=False)
    print("wrote src/enc_tests.zig")


main()
