#!/usr/bin/env python3
"""Generate src/x86_64/gp_table.zig: legacy-encoded x86-64 GP instruction forms from AsmJit's
ISA database (ref/asmjit/db/isa_x86.json).

Each DB form is expanded into concrete operand sizes (`rv` -> r16/r32/r64, `r8/m8` -> r8 and
m8, `immv` -> imm16/imm32, ...). Forms that need VEX/XOP/APX, segment/control/debug registers,
string-instruction implicit memory, moffs, relative branches (see the Assembler's branch API) or
far pointers are skipped. Run:

    python3 scripts/gen_gp_tables.py && python3 scripts/gen_gp_tests.py && zig build test
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIG_KEYWORDS = {"and", "or", "test", "int", "out", "in"}  # names quoted with @"..."

# operand token -> list of (kind, size) alternatives; size picks the expansion for rv/immv.
FIXED = {"al": "al", "ax": "ax", "eax": "eax", "rax": "rax", "cl": "cl", "dx": "dx", "1": "one"}


def zname(n):
    return f'@"{n}"' if n in ZIG_KEYWORDS or not re.match(r"^[a-z_][a-z0-9_]*$", n) else n


def expand_op(tok, osz):
    """Return list of kinds for one operand token at operand size osz (2/4/8), or None if unsupported."""
    t = tok.lstrip("~")
    if t.startswith("<"):
        return []  # implicit operand, not part of the API
    if t in FIXED:
        return [FIXED[t]]
    if t == "axv":
        return [{2: "ax", 4: "eax", 8: "rax"}[osz]]
    parts = t.split("/")
    out = []
    for p in parts:
        if p in ("r8", "r16", "r32", "r64"):
            out.append(p)
        elif p in ("rv", "ry"):
            out.append(f"r{osz * 8}")
        elif p in ("m8", "m16", "m32", "m64"):
            out.append(p)
        elif p in ("mv", "my"):
            out.append(f"m{osz * 8}")
        elif p == "m":
            out.append("m")
        elif p in ("imm8", "imms8", "immu8", "imm16", "immu16", "imm32", "imms32", "immu32", "imm64"):
            out.append({"imm8": "i8", "imms8": "s8", "immu8": "u8", "imm16": "i16", "immu16": "u16",
                        "imm32": "i32", "imms32": "s32", "immu32": "u32", "imm64": "i64"}[p])
        elif p == "immv":
            out.append({2: "i16", 4: "i32", 8: "s32"}[osz])
        else:
            return None
    return out


def sizes_for(inst, enc):
    toks = inst.replace(",", " ").split()
    if any(re.search(r"\b(rv|mv|immv|axv)\b", x) for x in toks):
        return [2, 4, 8]
    if any(re.search(r"\b(ry|my)\b", x) for x in toks):
        return [4, 8]
    return [0]


def parse_enc(enc, osz, uses_v):
    """Return dict(prefixes, rexw, opcode, plus_r, digit, modrm, imm) or None."""
    kind, _, rest = enc.partition(":")
    kind = kind.strip()
    if kind not in ("op", "m", "rm", "mr"):
        return None
    toks = rest.split()
    pre, op, rexw, plus, digit, modrm, imm = [], [], False, False, None, False, []
    for t in toks:
        if t in ("NP",):
            continue
        if t == "REX.W":
            rexw = True
        elif re.fullmatch(r"[0-9A-F]{2}", t):
            if not op and t in ("66", "67", "F2", "F3"):
                pre.append(int(t, 16))
            else:
                op.append(int(t, 16))
        elif re.fullmatch(r"[0-9A-F]{2}\+r", t):
            op.append(int(t[:2], 16))
            plus = True
        elif t == "/r":
            modrm = True
        elif re.fullmatch(r"/[0-7]", t):
            modrm, digit = True, int(t[1])
        elif t in ("ib", "iw", "id", "iq"):
            imm.append({"ib": 1, "iw": 2, "id": 4, "iq": 8}[t])
        elif t == "iv":
            imm.append({2: 2, 4: 4, 8: 4}[osz])
        else:
            return None
    if 67 in pre or 0x67 in pre or not op or len(imm) > 1 or len(op) > 3:
        return None
    if uses_v and osz == 2 and 0x66 not in pre:
        pre.insert(0, 0x66)
    if uses_v and osz == 8:
        rexw = True
    return dict(kind=kind, pre=pre, rexw=rexw, op=op, plus=plus, digit=digit, modrm=modrm, imm=imm[0] if imm else 0)


# AsmJit DB bugs: encodings corrected against the Intel SDM / LLVM.
ENC_OVERRIDE = {
    "shrd x:rv/mv, rv, imm8": "mr : 0F AC /r ib",  # DB has a spurious 66 prefix
    "lea w:r16, m": "rm : 66 8D /r",  # DB has 67 (address size) instead of the 16-bit operand size
}
# Mnemonics whose reg,reg form LLVM encodes as RM (reg field = first operand).
PREFER_RM = {"xchg"}

IMM_KINDS = {"i8", "s8", "u8", "i16", "u16", "i32", "s32", "u32", "i64", "one"}
FIXED_KINDS = {"al", "ax", "eax", "rax", "cl", "dx", "one"}


def forms():
    db = json.load(open(os.path.join(ROOT, "ref/asmjit/db/isa_x86.json")))
    out = {}
    for g in db["instructions"]:
        if g.get("category") != "GP" or g.get("ext"):
            continue
        for i in g["instructions"]:
          # Encodings per mode: "any" (both), "x64" (64-bit only), "x86" (32-bit only).
          if i.get("any") or i["inst"] in ENC_OVERRIDE:
              variants = [(ENC_OVERRIDE.get(i["inst"]) or i["any"], "any")]
          else:
              variants = [(i[m], m) for m in ("x64", "x86") if i.get(m)]
          for enc, mode in variants:
            inst = re.sub(r"^(\[[^\]]*\]\s*)+", "", i["inst"])
            mn, _, ops = inst.partition(" ")
            if not re.fullmatch(r"[a-z][a-z0-9]*", mn):
                continue
            toks = [re.sub(r"^[xXwWrR]:", "", o.strip()) for o in ops.split(",") if o.strip()]
            uses_v = any(re.search(r"(rv|mv|immv|axv|ry|my)", t) for t in toks)
            for osz in sizes_for(inst, enc):
                if mode == "x86" and osz == 8:
                    continue
                e = parse_enc(enc, osz, uses_v)
                if e is None:
                    continue
                # 16-bit-only spellings of zero-operand / immediate-only forms (pushf, popf,
                # push imm16): LLVM's plain mnemonic means the 64-bit default size.
                sized_ops = [t for t in toks if not t.startswith("<") and re.search(r"(^|/|~)(r|m|ax|al|eax|rax|cl|dx)", t)]
                if 0x66 in e["pre"] and not sized_ops:
                    continue
                alts = []
                ok = True
                for t in toks:
                    k = expand_op(t, osz)
                    if k is None:
                        ok = False
                        break
                    if k:
                        alts.append(k)
                if not ok:
                    continue
                # cartesian over r/m alternatives
                combos = [[]]
                for a in alts:
                    combos = [c + [k] for c in combos for k in a]
                for c in combos:
                    f = place(e, c)
                    if f:
                        f["ext"] = (i.get("ext") or "").split()
                        f["mode"] = mode
                        out.setdefault(mn, []).append(f)
    # Mnemonics for which only 32-bit-mode forms landed here (`dec` 48+r) but whose other forms
    # live in vec_table (category with `ext`): leave them all to vec_table.
    other = set()
    for g in db["instructions"]:
        for i in g["instructions"]:
            if i.get("any") or i.get("x64"):
                other.add(re.sub(r"^(\[[^\]]*\]\s*)+", "", i["inst"]).split(" ")[0].split("{")[0])
    for mn in list(out):
        if all(f["mode"] == "x86" for f in out[mn]) and mn in other:
            del out[mn]
    # de-dup, order: mr before rm (LLVM picks mr for reg,reg)
    for mn in out:
        seen, lst = set(), []
        first = "rm" if mn in PREFER_RM else "mr"
        for f in sorted(out[mn], key=lambda f: 0 if f["kind"] == first else 1):
            key = (f["mode"], tuple(f["ext"]), tuple(f["ops"]), tuple(f["pre"]), f["rexw"], tuple(f["op"]), f["digit"], f["imm"], f["kind"])
            if key not in seen:
                seen.add(key)
                lst.append(f)
        out[mn] = lst
    return out


def place(e, ops):
    f = dict(e, ops=ops, reg=None, rm=None, opreg=None)
    gen = [i for i, k in enumerate(ops) if k not in IMM_KINDS and k not in FIXED_KINDS]
    imm_ops = [i for i, k in enumerate(ops) if k in IMM_KINDS - {"one"}]
    if len(imm_ops) != (1 if e["imm"] else 0):
        return None
    if e["imm"] and imm_ops:
        k = ops[imm_ops[0]]
        need = {"i8": 1, "s8": 1, "u8": 1, "i16": 2, "u16": 2, "i32": 4, "s32": 4, "u32": 4, "i64": 8}[k]
        if need != e["imm"]:
            return None
    if e["kind"] == "op":
        if e["modrm"]:
            return None
        if e["plus"]:
            if not gen or ops[gen[0]][0] != "r":
                return None
            f["opreg"] = gen[0]
            if len(gen) > 1:
                return None
        elif gen:
            return None
    elif e["kind"] == "m":
        if not e["modrm"] or len(gen) != 1:
            return None
        f["rm"] = gen[0]
        if e["digit"] is None:
            f["digit"] = 0  # e.g. setcc "/r": reg field ignored, LLVM encodes 0
    else:
        if not e["modrm"] or e["digit"] is not None or len(gen) != 2:
            return None
        a, b = gen
        f["reg"], f["rm"] = (a, b) if e["kind"] == "rm" else (b, a)
        if ops[f["reg"]][0] != "r":
            return None
    return f


def zig_form(f):
    ops = ", ".join("." + k for k in f["ops"])
    opt = lambda v: "null" if v is None else str(v)
    return (f'.{{ .ops = &.{{{ops}}}, .pre = &.{{{", ".join(hex(b) for b in f["pre"])}}}, '
            f'.op = &.{{{", ".join(hex(b) for b in f["op"])}}}, .rexw = {str(f["rexw"]).lower()}, '
            f'.plus_r = {str(f["plus"]).lower()}, .modrm = {str(f["modrm"]).lower()}, .digit = {opt(f["digit"])}, '
            f'.imm = {f["imm"]}, .reg = {opt(f["reg"])}, .rm = {opt(f["rm"])}, .opreg = {opt(f["opreg"])}, '
            f'.features = &.{{{", ".join(feat(x) for x in f["ext"])}}}'
            + ("" if f["mode"] == "any" else f', .mode = .{f["mode"]}') + ' }')


def feat(x):
    return "." + (f'@"{x}"' if x[0].isdigit() else x)


def main():
    fs = forms()
    lines = ["// Generated by scripts/gen_gp_tables.py from ref/asmjit/db/isa_x86.json. Do not edit.",
             f"// {sum(len(v) for v in fs.values())} forms, {len(fs)} mnemonics.",
             'const Form = @import("gp.zig").Form;', "", "pub const Mnemonic = enum {"]
    lines += [f"    {zname(m)}," for m in sorted(fs)]
    lines += ["};", "", "pub const table = struct {"]
    for m in sorted(fs):
        lines.append(f"    pub const {zname(m)} = [_]Form{{")
        lines += [f"        {zig_form(f)}," for f in fs[m]]
        lines.append("    };")
    lines.append("};")
    open(os.path.join(ROOT, "src/x86_64/gp_table.zig"), "w").write("\n".join(lines) + "\n")
    json.dump(fs, open(os.path.join(ROOT, "scripts/.gp_forms.json"), "w"))
    print(f"{sum(len(v) for v in fs.values())} forms, {len(fs)} mnemonics")


if __name__ == "__main__":
    main()
