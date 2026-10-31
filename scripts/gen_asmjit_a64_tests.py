#!/usr/bin/env python3
"""Port AsmJit's AArch64 assembler tests (asmjit_test_assembler_a64.cpp) to pyrlyn-jit's table-driven
`a64` encoder. Output: src/asmjit_a64_tests.zig + bench/asmjit_a64_unsupported.txt.

    python3 scripts/gen_asmjit_a64_tests.py [path/to/asmjit]   # default ../asmjit-loc
"""
import re, sys, collections, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
ref = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root.parent / "asmjit-loc")
src = (ref / "asmjit-testing/tests/asmjit_test_assembler_a64.cpp").read_text()
mnems = set(re.findall(r'^    @"([\w.]+)",$', (root / "src/aarch64/a64_table.zig").read_text().split("pub const Mnemonic = enum {")[1].split("};")[0], re.M))
COND = dict(eq=0, ne=1, hs=2, cs=2, lo=3, cc=3, mi=4, pl=5, vs=6, vc=7, hi=8, ls=9, ge=10, lt=11, gt=12, le=13, al=14, nv=15)
ARR = {"b8": "a8b", "b16": "a16b", "h4": "a4h", "h8": "a8h", "s2": "a2s", "s4": "a4s", "d1": "a1d", "d2": "a2d", "q1": "a1q", "b4": "a4b", "h2": "a2h", "b2": "a2b", "s1": "a1s"}
SHIFTS = {"lsl", "lsr", "asr", "ror", "msl", "uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"}

class Unsup(Exception):
    pass

KNOWN = set(l.strip() for l in (root / "bench/asmjit_a64_known_failures.txt").read_text().splitlines() if l.strip()) \
    if (root / "bench/asmjit_a64_known_failures.txt").exists() else set()
known = []

def _predicates():
    """Predicate::<NS>::k<Name> values from AsmJit's a64_globals.h (AT/DC/IC/TLBI/BTI/SysReg/...)."""
    hdr = (root / "ref/asmjit/asmjit/arm/a64_globals.h")
    if not hdr.exists(): hdr = pathlib.Path.home() / "Grok/asmjit-loc/asmjit/arm/a64_globals.h"
    out, ns = {}, None
    for line in hdr.read_text().splitlines():
        m = re.match(r"\s*namespace (\w+) \{", line)
        if m: ns = m.group(1); continue
        m = re.match(r"\s*k(\w+)\s*=\s*(encode\(([^)]*)\)|0x[0-9A-Fa-f]+u?|\d+u?)\s*,?", line)
        if not m or ns is None: continue
        if m.group(3) is not None:
            a = [int(x.strip().replace("0b", ""), 2) for x in m.group(3).split(",")]
            if len(a) == 4: v = a[0] << 11 | a[1] << 7 | a[2] << 3 | a[3]
            elif len(a) == 5: v = a[0] << 14 | a[1] << 11 | a[2] << 7 | a[3] << 3 | a[4]
            elif len(a) == 2: v = a[0] << 3 | a[1]
            else: continue
        else: v = int(m.group(2).rstrip("u"), 0)
        out.setdefault(f"Predicate::{ns}::k{m.group(1)}", v)
    return out
PRED = _predicates()

def split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(": depth += 1
        if ch == ")": depth -= 1
        if ch == "," and depth == 0: out.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def num(t):
    t = t.strip()
    m = re.fullmatch(r"(-?)(0x[0-9a-fA-F]+|\d+)(u|ull|ll|LL|ULL|U)?", t)
    if not m: return None
    v = int(m.group(2), 0)
    return -v if m.group(1) else v

def reg(t):
    if t in ("wzr", "xzr", "wsp", "sp"): return f"a64.{t}"
    m = re.fullmatch(r"([wx])(\d+)", t)
    if m: return f"a64.{m.group(1)}({m.group(2)})"
    m = re.fullmatch(r"([bhsdq])(\d+)", t)
    if m: return f"a64.scalar(.{m.group(1)}, {m.group(2)})"
    m = re.fullmatch(r"v(\d+)\.(\w+)\((\d*)\)", t)
    if m:
        i, a, ix = m.groups()
        if a in ("b", "h", "s", "d", "q") and ix: return f"a64.elem({i}, .{a}, {ix})"
        if a in ARR and not ix: return f"a64.vec({i}, .{ARR[a]})"
        if a in ARR and ix: return f"a64.elemArr({i}, .{ARR[a]}, {ix})"
    return None

def operand(t):
    t = t.strip()
    r = reg(t)
    if r: return r
    n = num(t)
    if n is not None:
        if n >= 2**63: n -= 2**64  # 64-bit patterns as two's complement
        return f"a64.imm({n})"
    if re.fullmatch(r"-?\d+\.\d+(f)?", t): return f"a64.fimm({t.rstrip('f')})"
    m = re.fullmatch(r"CondCode::k(\w+)", t)
    if m: return f"a64.cond({COND[m.group(1).lower()]})"
    if t == "L0": return "a64.rel(0)"
    if t in PRED and not t.startswith("Predicate::PRFOp"): return f"a64.imm({PRED[t]})"
    m = re.fullmatch(r"Predicate::PRFOp::kP(LD|LI|ST)L(\d)(KEEP|STRM)", t)
    if m:
        return f"a64.imm({ {'LD': 0, 'LI': 1, 'ST': 2}[m.group(1)] << 3 | (int(m.group(2)) - 1) << 1 | (m.group(3) == 'STRM') })"
    m = re.fullmatch(r"(\w+)\((.*)\)", t)
    if m and m.group(1) in SHIFTS:
        return f"a64.shift(.{m.group(1)}, {num(m.group(2))})"
    if m and m.group(1) in ("ptr", "ptr_pre", "ptr_post"):
        a = split_args(m.group(2))
        base = reg(a[0])
        if not base: raise Unsup("mem-base")
        if len(a) == 1: return f"a64.ptr({base}, 0)"
        off = num(a[1])
        if m.group(1) == "ptr":
            if off is not None and len(a) == 2: return f"a64.ptr({base}, {off})"
            idx = reg(a[1])
            if not idx: raise Unsup("mem-index")
            ext = operand(a[2]) if len(a) > 2 else "null"
            return f"a64.ptrReg({base}, {idx}, {ext})"
        if off is not None: return f"a64.{'ptrPre' if m.group(1) == 'ptr_pre' else 'ptrPost'}({base}, {off})"
        idx = reg(a[1])
        if m.group(1) == "ptr_post" and idx: return f"a64.ptrPostReg({base}, {idx})"
        raise Unsup("mem")
    raise Unsup("operand:" + t.split("(")[0][:30])

cases, unsup, unsup_cases, total = [], collections.Counter(), [], 0
for hexs, body in re.findall(r'TEST_INSTRUCTION\("([0-9A-F]+)"\s*,\s*(.*?)\);', src):
    total += 1
    try:
        m = re.fullmatch(r"(\w+)\((.*)\)", body.strip())
        if not m: raise Unsup("syntax")
        mn = m.group(1)
        args = split_args(m.group(2))
        ops = [operand(a) for a in args]
        mc = re.fullmatch(r"b_(\w+)", mn)
        if mc and mc.group(1) in COND:
            c = COND[mc.group(1)]
            mn = "b"
            if c != 14: mn = "b_cond"; ops = [f"a64.cond({c})"] + ops
        if mn == "b" and len(ops) == 2 and ops[0].startswith("a64.cond("):
            if ops[0] == "a64.cond(14)": ops = ops[1:]
            else: mn = "b_cond"
        if mn in ("adr", "adrp") and len(ops) == 2 and ops[1].startswith("a64.imm("):
            ops[1] = ops[1].replace("a64.imm(", "a64.rel(")  # target address, code starts at 0
        if mn not in mnems: mn = mn.rstrip("_")
        if mn not in mnems: raise Unsup("mnemonic:" + mn)
        word = int.from_bytes(bytes.fromhex(hexs), "little")
        txt = (hexs + " " + body.strip()).replace("\\", "\\\\").replace('"', '\\"')
        key = (hexs + " " + body.strip())
        if key in KNOWN:
            known.append(f'    xfail += checkQuiet(0x{word:08X}, .@"{mn}", &.{{ {", ".join(ops)} }}, "{txt}");')
            continue
        cases.append(f'    fails += check(0x{word:08X}, .@"{mn}", &.{{ {", ".join(ops)} }}, "{txt}");')
    except Unsup as e:
        unsup[str(e).split(":")[0]] += 1
        unsup["  " + str(e)] += 1
        unsup_cases.append(f"{e}: {hexs} {body}")

out_known = known
out = [f"//! GENERATED by scripts/gen_asmjit_a64_tests.py — do not edit. Expected words from AsmJit's",
       f"//! asmjit_test_assembler_a64.cpp: {total} cases, {len(cases)} passing, {len(known)} known failures, {total - len(cases) - len(known)} unsupported.",
       'const std = @import("std");', 'const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;',
       'const a64 = @import("aarch64/a64.zig");', "",
       "fn check(expected: u32, comptime mn: a64.Mnemonic, ops: []const a64.Op, text: []const u8) u32 {",
       "    const r = a64.encodeMnIdx(mn, ops) orelse {",
       '        std.debug.print("NoMatchingForm: {s}\\n", .{text});', "        return 1;", "    };",
       "    if (r.word == expected) return 0;",
       '    std.debug.print("encoding mismatch: {s}: got {X:0>8} form={s}#{d}\\n", .{ text, @byteSwap(r.word), @tagName(mn), r.idx });',
       "    return 1;", "}", "",
       "fn checkQuiet(expected: u32, comptime mn: a64.Mnemonic, ops: []const a64.Op, text: []const u8) u32 {",
       "    _ = text;",
       "    const r = a64.encodeMnIdx(mn, ops) orelse return 1;",
       "    return @intFromBool(r.word != expected);", "}", ""]
if out_known: out += ["// Known failures (bench/asmjit_a64_known_failures.txt): encoder gaps still being closed.",
        "// This test reports how many still fail and errors if any of them start passing (so the list stays honest).",
        'test "AsmJit a64 known failures" {', "    var xfail: u32 = 0;"] + out_known + \
       [f"    if (xfail != {len(out_known)}) {{",
        f'        std.debug.print("a64 known failures: {{d}} of {len(out_known)} still fail; update bench/asmjit_a64_known_failures.txt\\n", .{{xfail}});',
        "        return error.KnownFailuresChanged;", "    }", "}", ""]
N = 400
for i in range(0, len(cases), N):
    out += [f'test "AsmJit a64 assembler tests ({i}..{min(i + N, len(cases))})" {{', "    var fails: u32 = 0;"] + cases[i:i + N] + \
           ["    try std.testing.expectEqual(@as(u32, 0), fails);", "}", ""]
(root / "src/asmjit_a64_tests.zig").write_text("\n".join(out))
rep = [f"total {total}, passing {len(cases)}, known failures {len(known)}, unsupported {total - len(cases) - len(known)}"] + [f"{v:6} {k}" for k, v in unsup.most_common() if not k.startswith("  ")]
rep += ["", "top unsupported details:"] + [f"{v:6} {k.strip()}" for k, v in unsup.most_common() if k.startswith("  ")][:60]
rep += ["", "cases:"] + unsup_cases
(root / "bench/asmjit_a64_unsupported.txt").write_text("\n".join(rep) + "\n")
print(rep[0])
