#!/usr/bin/env python3
"""Differential SVE/SME tests: for every SVE form in the generated table, synthesize operands,
assemble the text with the system clang (LLVM's AArch64 assembler is the oracle) and record
the expected word. Writes src/a64_sve_tests.zig + bench/a64_sve_coverage.txt.

    python3 scripts/gen_sve_tests.py
"""
import collections, itertools, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_a64_tables as g

ROOT = g.ROOT
MARCH = "armv9.4-a+sve2+sme2+sve2-aes+sve2-sha3+sve2-sm4+sve2-bitperm+f64mm+i8mm+bf16+sve-b16b16+sme-f64f64+sme-i16i64"
SEP = ".inst 0xd503201f"  # data word between candidates (keeps movprfx checks local)

def ets(lst):
    return [e for e in (lst or []) if e != "~"]

IMMS = {"imm": [1, 2, 3, 7, 16], "pattern": [31, 5], "rot2": [90, 270], "rot1": [90, 270],
        "n": [8], "imm1": [1], "imm2": [1]}

def cases_for(mn, f):
    """Yield (asm text, zig ops) pairs for form f."""
    specs = f["ops"]
    choices = [None]
    sizes = ets(f.get("zt")) or [None]
    sizes_a = ets(f.get("zta")); sizes_b = ets(f.get("ztb"))
    pairs = list(zip(f.get("zta") or [], f.get("ztb") or []))
    pairs = [(a, b) for a, b in pairs if a != "~" and b != "~"] or [(None, None)]
    out = []
    for t in sizes:
        for ta, tb in pairs:
            for variant in range(2):
                txt, zig, cnt = [], [], itertools.count(1)
                tied = {}
                def reg(s=None):
                    f = s.get("f") if s else None
                    if f and f in tied: return tied[f]
                    r = next(cnt)
                    if f: tied[f] = r
                    return r
                ok = True
                for s in specs:
                    k = s["k"]
                    if k in ("zt", "zta", "ztb", "zfix"):
                        e = {"zt": t, "zta": ta, "ztb": tb, "zfix": (s.get("et") or "").upper()}[k]
                        if not e: ok = False; break
                        r = reg(s)
                        txt.append(f"z{r}.{e.lower()}"); zig.append(f"a64.zreg({r}, .{e.lower()})")
                    elif k == "zelem":
                        r = reg(s) % 8; idx = s.get("fixed", variant)
                        txt.append(f"z{r}.{s['et']}[{idx}]"); zig.append(f"a64.zelem({r}, .{s['et']}, {idx})")
                    elif k in ("pg", "pgm", "pgz", "pgmz"):
                        q = {"pg": "", "pgm": "/m", "pgz": "/z", "pgmz": "/m" if variant == 0 else "/z"}[k]
                        txt.append(f"p1{q}"); zig.append({"": "a64.preg(1, .none)", "/m": "a64.pmerge(1)", "/z": "a64.pzero(1)"}[q])
                    elif k in ("pt", "pfix"):
                        e = (t if k == "pt" else s["et"]) or ""
                        if not e: ok = False; break
                        r = 2 + reg(s) % 6
                        txt.append(f"p{r}.{e.lower()}"); zig.append(f"a64.preg({r}, .{e.lower()})")
                    elif k in ("w", "x", "wsp", "xsp", "rext"):
                        if s.get("fixed") and not s.get("f"): txt.append(f"x{s['fixed']}"); zig.append(f"a64.x({s['fixed']})"); continue
                        r = reg(s); x = k in ("x", "xsp") or (k == "rext" and variant == 0)
                        txt.append(("x" if x else "w") + str(r)); zig.append(f"a64.{'x' if x else 'w'}({r})")
                    elif k in ("sb", "sh", "ss", "sd", "sq", "sany"):
                        kind = k[1] if k != "sany" else (t or "d").lower()
                        r = reg(s)
                        txt.append(f"{kind}{r}"); zig.append(f"a64.scalar(.{kind}, {r})")
                    elif k == "imm":
                        name = s.get("f")
                        if s.get("opt") and variant == 0 and name in ("imm", "pattern") and len([x for x in specs if x["k"] == "imm"]) > 1:
                            continue
                        vals = IMMS.get(name, [1])
                        v = vals[variant % len(vals)] if name != "imm" else vals[variant]
                        if name == "imm" and s.get("opt"): txt.append(f"mul #{v + 1}"); zig.append(f"a64.imm({v + 1})"); continue
                        txt.append(f"#{v}"); zig.append(f"a64.imm({v})")
                    elif k == "immfix":
                        txt.append(f"#{s['fixed']}"); zig.append(f"a64.imm({s['fixed']})")
                    elif k == "zelemt":
                        if not t: ok = False; break
                        r = reg(s); idx = variant
                        txt.append(f"z{r}.{t.lower()}[{idx}]"); zig.append(f"a64.zelem({r}, .{t.lower()}, {idx})")
                    elif k == "fimm":
                        v = [0.5, 1.0][variant]
                        txt.append(f"#{v}"); zig.append(f"a64.fimm({v})")
                    elif k == "shift":
                        if variant == 0: continue
                        txt.append("lsl #8"); zig.append("a64.shift(.lsl, 8)")
                    else:
                        ok = False; break
                if ok: out.append((f"{mn} " + ", ".join(txt), zig))
    return out

def assemble(lines):
    """-> {line: word} for the lines clang accepts."""
    good = list(dict.fromkeys(lines))
    with tempfile.TemporaryDirectory() as d:
        src, obj = os.path.join(d, "t.s"), os.path.join(d, "t.o")
        while True:
            open(src, "w").write("".join(f"{l}\n{SEP}\n" for l in good))
            r = subprocess.run(["clang", "-c", "-target", "aarch64-apple-macos", "-march=" + MARCH, src, "-o", obj], capture_output=True, text=True)
            bad = {int(m.group(1)) for m in re.finditer(r"t\.s:(\d+):\d+: error", r.stderr)}
            if r.returncode == 0: break
            if not bad: raise SystemExit(r.stderr[:2000])
            good = [l for i, l in enumerate(good) if 2 * i + 1 not in bad]
        dis = subprocess.run(["objdump", "-d", obj], capture_output=True, text=True).stdout
        words = [int(m.group(1), 16) for m in re.finditer(r"^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", dis, re.M)]
    assert len(words) == 2 * len(good), (len(words), len(good))
    return {l: words[2 * i] for i, l in enumerate(good)}

def main():
    table = g.main(write=False)
    cands = []
    nforms = 0
    for mn, forms in table.items():
        for f in forms:
            if f.get("zt") is None and f.get("zta") is None and not any(o["k"].startswith(("z", "p")) for o in f["ops"]):
                continue
            nforms += 1
            for txt, zig in cases_for(mn.replace("_cond", ""), f):
                cands.append((mn, txt, zig, f["inst"]))
    words = assemble([c[1] for c in cands])
    seen, cases, rejected = set(), [], collections.Counter()
    for mn, txt, zig, inst in cands:
        if txt not in words: rejected[inst] += 1; continue
        if txt in seen: continue
        seen.add(txt)
        cases.append(f'    fails += check(0x{words[txt]:08X}, .@"{mn}", &.{{ {", ".join(zig)} }}, "{txt}");')
    known = set()
    kp = os.path.join(ROOT, "bench/a64_sve_known_failures.txt")
    if os.path.exists(kp): known = {l.strip() for l in open(kp) if l.strip() and not l.startswith("#")}
    live = [c for c in cases if c.rsplit('"', 2)[-2] not in known]
    xf = [c.replace("fails += check(", "xfail += checkQuiet(") for c in cases if c.rsplit('"', 2)[-2] in known]
    out = ["//! GENERATED by scripts/gen_sve_tests.py - do not edit. Expected words from LLVM's AArch64",
           f"//! assembler (system clang, -march={MARCH}): {len(cases)} cases from {nforms} SVE/SME forms.",
           'const std = @import("std");', 'const a64 = @import("aarch64/a64.zig");', "",
           "fn check(expected: u32, comptime mn: a64.Mnemonic, ops: []const a64.Op, text: []const u8) u32 {",
           "    const r = a64.encodeMnIdx(mn, ops) orelse {", '        std.debug.print("NoMatchingForm: {s}\\n", .{text});', "        return 1;", "    };",
           "    if (r.word == expected) return 0;",
           '    std.debug.print("encoding mismatch: {s}: want {X:0>8} got {X:0>8} form={s}#{d}\\n", .{ text, expected, r.word, @tagName(mn), r.idx });',
           "    return 1;", "}", "",
           "fn checkQuiet(expected: u32, comptime mn: a64.Mnemonic, ops: []const a64.Op, text: []const u8) u32 {",
           "    _ = text;", "    const r = a64.encodeMnIdx(mn, ops) orelse return 1;", "    return @intFromBool(r.word != expected);", "}", ""]
    N = 400
    for i in range(0, len(live), N):
        out += [f'test "SVE/SME vs LLVM ({i}..{min(i + N, len(live))})" {{', "    var fails: u32 = 0;"] + live[i:i + N] + \
               ["    try std.testing.expectEqual(@as(u32, 0), fails);", "}", ""]
    if xf:
        out += ['test "SVE/SME known failures (bench/a64_sve_known_failures.txt)" {', "    var xfail: u32 = 0;"] + xf + \
               [f"    if (xfail != {len(xf)}) return error.KnownFailuresChanged;", "}", ""]
    open(os.path.join(ROOT, "src/a64_sve_tests.zig"), "w").write("\n".join(out))
    rep = [f"SVE/SME forms: {nforms}; LLVM-verified cases: {len(cases)} (passing {len(live)}, known failures {len(xf)})",
           f"forms with no case accepted by LLVM: {len(rejected)} (DB syntax LLVM does not accept; these forms are verified by the hand-spelled LLVM cases in src/a64_sve_extra_tests.zig)", ""]
    rep += [f"{v:4} {k}" for k, v in sorted(rejected.items())]
    open(os.path.join(ROOT, "bench/a64_sve_coverage.txt"), "w").write("\n".join(rep) + "\n")
    print(rep[0]); print(rep[1])

if __name__ == "__main__":
    main()
