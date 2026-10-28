#!/usr/bin/env python3
"""Generate src/x86_64/vec_table.zig: every non-GP x86-64 form of AsmJit's ISA database
(SSE, AVX, AVX-512, AVX10.2, MMX/3DNow!, x87, AVX-512 mask, AMX, crypto, ...) as data for the
table-driven encoder in src/x86_64/vec.zig (legacy, VEX, XOP and EVEX encodings).

Vector-length placeholders are expanded per L (`xy`, `xyz`, `xxy`, `mxyz`, ...); `r/m`
alternatives become separate register and memory forms; implicit operands (`<xmm0>`, `k+1`)
are dropped. APX-only encodings are skipped.

    python3 scripts/gen_vec_tables.py        # reads ref/asmjit/db/isa_x86.json
"""
import json, os, re, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_CATS = {"GP"}

VEC = ["xmm", "ymm", "zmm"]
MSZ = [16, 32, 64]

def expand_tok(t, li):
    """Map one operand token to (kind, size, extra) for vector length index li (0/1/2 or None)."""
    t = t.lstrip("~")
    t = re.sub(r"\[\d+:\d+\]$", "", t)
    t = re.sub(r"@\d+$", "", t)
    if t in ("xy", "xyz"): return (VEC[li], 0)
    if t == "xxy": return (["xmm", "xmm", "ymm"][li], 0)
    if t in ("xxx",): return ("xmm", 0)
    if t in ("mxy", "mxyz"): return ("mem", MSZ[li])
    if t == "mxxy": return ("mem", [8, 16, 32][li])
    if t == "mxxx": return ("mem", [4, 8, 16][li])
    if t in ("xmm", "ymm", "zmm", "mm", "k", "tmm", "st0", "sti", "r8", "r16", "r32", "r64", "ax"): return (t, 0)
    far = {"m16_16": 4, "m16_32": 6, "m16_64": 10}
    if t in far: return ("mem", far[t])
    m = re.fullmatch(r"m(\d+)(?:_\w+)?(?:\(.*\))?", t)
    if m: return ("mem", int(m.group(1)) // 8)
    if t == "m(ds:r64)": return ("maddr", 0)
    if t in ("m", "tmem") or re.fullmatch(r"m\(\w+:\w+\)", t): return ("mem", 0)
    m = re.fullmatch(r"vm(32|64)([xyz])", t)
    if m: return ("v" + m.group(2), int(m.group(1)) // 8)
    if t in ("imm8", "imm4", "ib", "imms8", "immu8"): return ("imm", 1)
    if t in ("imm16", "immu16", "imms16"): return ("imm", 2)
    if t in ("imm32", "imms32", "immu32"): return ("imm", 4)
    if t in ("imm64",): return ("imm", 8)
    if t in ("eax", "rax", "al", "cl", "dx"): return (t, 0)
    raise KeyError(t)

def parse_enc(e):
    toks = e.split()
    d = dict(enc="legacy", pre=[], rexw=False, map=0, op=[], digit=None, imm=0, plus=False,
             is4=False, modrm=False, only=None, pp=0, w=2, l=None, lvar=None)
    if toks[0].startswith(("VEX.", "EVEX.", "XOP.")):
        parts = toks[0].split(".")
        d["enc"] = parts[0].lower()
        for p in parts[1:]:
            if p in ("128", "L0", "LZ", "LLZ"): d["l"] = 0
            elif p in ("256", "L1"): d["l"] = 1
            elif p == "512": d["l"] = 2
            elif p == "LIG": d["l"] = 0
            elif p == "Lxy": d["lvar"] = [0, 1]
            elif p == "xyz": d["lvar"] = [0, 1, 2]
            elif p in ("NP", "P0"): d["pp"] = 0
            elif p == "66": d["pp"] = 1
            elif p == "F3": d["pp"] = 2
            elif p == "F2": d["pp"] = 3
            elif p == "0F": d["map"] = 1
            elif p == "0F38": d["map"] = 2
            elif p == "0F3A": d["map"] = 3
            elif p.startswith("MAP"): d["map"] = int(p[3:], 16)
            elif p == "W0": d["w"] = 0
            elif p == "W1": d["w"] = 1
            elif p == "WIG": d["w"] = 2
            elif p == "Wy": d["w"] = 3
            else: raise KeyError(p)
        toks = toks[1:]
    else:
        # legacy: prefixes, REX.W, escape bytes, opcode
        while toks and toks[0] in ("NP", "66", "67", "F2", "F3", "REX.W", "9B", "NFx", "NOREP"):
            t = toks.pop(0)
            if t in ("NFx", "NOREP"): continue
            if t == "REX.W": d["rexw"] = True
            elif t == "NP": pass
            elif t == "9B": d["pre"].append(0x9B)
            else: d["pre"].append(int(t, 16))
        if toks[:2] == ["0F", "38"]: d["map"] = 2; toks = toks[2:]
        elif toks[:2] == ["0F", "3A"]: d["map"] = 3; toks = toks[2:]
        elif toks[:2] == ["0F", "0F"]: d["map"] = 4; toks = toks[2:]  # 3DNow!: 0F 0F /r ib(opcode)
        elif toks[:1] == ["0F"]: d["map"] = 1; toks = toks[1:]
    for t in toks:
        if re.fullmatch(r"[0-9A-Fa-f]{2}", t): d["op"].append(int(t, 16))
        elif re.fullmatch(r"[0-9A-F]{2}\+i", t): d["op"].append(int(t[:2], 16)); d["plus"] = True
        elif t == "/r": d["modrm"] = True
        elif re.fullmatch(r"/[0-7]", t): d["modrm"] = True; d["digit"] = int(t[1])
        elif t == "/is4": d["is4"] = True
        elif t in ("ib",): d["imm"] += 1
        elif t == "iw": d["imm"] += 2
        elif t == "id": d["imm"] += 4
        elif m := re.fullmatch(r"(!\(11\)|11):(rrr|[01]{3}):(bbb|[01]{3})", t):
            d["modrm"] = True
            d["only"] = "mem" if m.group(1).startswith("!") else "reg"
            if m.group(2) != "rrr": d["digit"] = int(m.group(2), 2)
            if m.group(3) != "bbb": d["sib"] = True
        else: raise KeyError(t)
    return d

# AsmJit's generated C++ instruction table (asmjit/x86/x86_inst_db.cpp) is what AsmJit's own
# encoder uses; the JSON has a number of stale prefix/map/W fields. For every JSON form whose
# opcode byte matches the C++ main/alt opcode, pp/map/W and the disp8*N rule come from C++.
PFX = {"00": 0, "66": 1, "F3": 2, "F2": 3}
def load_cpp():
    src = open(os.path.join(ROOT, "ref/asmjit/asmjit/x86/x86_inst_db.cpp")).read()
    out = {}
    pat = re.compile(r"([VEO])\(([0-9A-FX_MAP]+),([0-9A-F]{2}),(\w),(\w),(\w),(\w),(\w),\s*(\w+)\s*\)")
    for m in re.finditer(r"INST\((\w+)\s*,\s*(\w+)\s*,(.*?)\),\s*//", src):
        name = m.group(1).lower()
        ops = []
        for o in pat.finditer(m.group(3)):
            kind, pre, op, modo, ll, w, ew, n, tt = o.groups()
            pp = PFX.get(pre[:2]); mp = pre[2:]
            mapv = {"0F00": 1, "0F38": 2, "0F3A": 3, "0000": 0, "0F01": 1, "0F0F": 4, "MAP5": 5, "MAP6": 6}.get(mp)
            if pre.startswith("XOP_M"): pp = 0; mapv = int(pre[5:], 16)
            ops.append(dict(kind=kind, pp=pp, map=mapv, op=int(op, 16), digit=None if modo == "_" else int(modo),
                            w=w, ew=ew, n=n, tt=tt))
        common = int(m.group(3).split(",")[-2])
        out[name] = (m.group(2), ops, common)
    prefer = set()
    for cm in re.finditer(r"\{\s*([^{}]*?F\(PreferEvex\)[^{}]*?)\}\s*,\s*//\s*#(\d+)", src):
        prefer.add(int(cm.group(2)))
    for k_, v_ in list(out.items()):
        out[k_] = (v_[0], v_[1], v_[2] in prefer)
    return out

def cpp_fix(cpp, name, d, form_enc, siblings):
    ent = cpp.get(name)
    if not ent or d["enc"] == "legacy": return None
    # Same opcode byte with different prefixes in the JSON (vmovq 66/F3 7E): leave it alone.
    if len({(x["pp"], x["map"]) for x in siblings if x["op"][-1:] == d["op"][-1:] and x["enc"] == d["enc"]}) > 1: return None
    for o in ent[1]:
        if o["op"] != d["op"][-1] or o["map"] is None or o["kind"] == "O": continue
        if o["digit"] is not None and d["digit"] is not None and o["digit"] != d["digit"]: continue
        if o["kind"] == "E" and form_enc != "evex": continue
        return dict(o, idx=ent[1].index(o))
    return None

def main():
    cpp = load_cpp()
    db = json.load(open(os.path.join(ROOT, "ref/asmjit/db/isa_x86.json")))
    sib_encs = collections.defaultdict(list)
    for cat in db["instructions"]:
        if cat["category"] in SKIP_CATS: continue
        for ins in cat["instructions"]:
            enc = ins.get("any") or ins.get("x64")
            if not enc: continue
            try: sib_encs[re.sub(r"^\[[^\]]*\]\s*", "", ins["inst"]).split()[0]].append(parse_enc(enc.split(":", 1)[1].strip()))
            except KeyError: pass
    db = json.load(open(os.path.join(ROOT, "ref/asmjit/db/isa_x86.json")))
    table = collections.OrderedDict()
    skipped = collections.Counter()
    gp_mn = set(re.findall(r"^    (\w+|@\"\w+\"),$", open(os.path.join(ROOT, "src/x86_64/gp_table.zig")).read().split("pub const Mnemonic = enum {")[1].split("};")[0], re.M))
    gp_mn = {m.strip('@"') for m in gp_mn}
    def gp_variants(ins):
        s = ins["inst"]
        if not re.search(r"\b(rv|mv|immv|axv|dxv)\b", s): return [(ins, None)]
        out = []
        for sz, r, m, im, ax in ((2, "r16", "m16", "imm16", "ax"), (4, "r32", "m32", "imm32", "eax"), (8, "r64", "m64", "imm32", "rax")):
            t = re.sub(r"\brv\b", r, s); t = re.sub(r"\bmv\b", m, t); t = re.sub(r"\bimmv\b", im, t)
            t = re.sub(r"\baxv\b", ax, t); t = re.sub(r"\bdxv\b", {"ax": "dx", "eax": "edx", "rax": "rdx"}[ax], t)
            out.append((dict(ins, inst=t), sz))
        return out
    for cat in db["instructions"]:
        if cat["category"] in SKIP_CATS and cat["category"] != "GP": continue
        expanded = []
        for ins in cat["instructions"]:
            if cat["category"] == "GP":
                nm = re.sub(r"^\[[^\]]*\]\s*", "", ins["inst"]).split()[0]
                if nm in gp_mn: continue
                expanded += gp_variants(ins)
            else:
                expanded.append((ins, None))
        for ins, gsz in expanded:
            enc = ins.get("any") or ins.get("x64")
            if not enc: skipped["apx/x86-only"] += 1; continue
            role, e = [x.strip() for x in enc.split(":", 1)]
            role = role.rstrip("_")
            s = re.sub(r"^\[[^\]]*\]\s*", "", ins["inst"])
            mods = set(re.findall(r"\{([^}]*)\}", s))
            s = re.sub(r"\s*\{[^}]*\}", "", s)
            name, _, opstr = s.partition(" ")
            raw_ops = [re.sub(r"^[A-Za-z?]+:", "", o.strip()) for o in opstr.split(",")] if opstr.strip() else []
            raw_ops = [o for o in raw_ops if not o.lstrip("~").startswith("<") and o != "k+1" and o != "1"]
            try:
                d = parse_enc(e)
            except KeyError as ex:
                skipped["enc " + str(ex)] += 1; continue
            if gsz == 2 and d["enc"] == "legacy" and 0x66 not in d["pre"]: d["pre"] = [0x66] + d["pre"]
            if gsz == 8 and d["enc"] == "legacy": d["rexw"] = True
            fx = cpp_fix(cpp, name, d, d["enc"], sib_encs[name])
            if fx:
                if fx["pp"] is not None: d["pp"] = fx["pp"]
                if fx["map"] is not None and (d["enc"] != "xop") == (fx["map"] < 8): d["map"] = fx["map"]
                wv = fx["ew"] if d["enc"] == "evex" else fx["w"]
                if wv in ("0", "1"): d["w"] = int(wv)
                elif wv == "I" and d["w"] != 1: d["w"] = 2
                # Role from the C++ encoding name, e.g. VexRmMr_Lx: main opcode `rm`, alt `mr`.
                parts = re.findall(r"[A-Z][a-z]*", cpp[name][0].split("_")[0])
                parts = [p_.lower() for p_ in parts if re.fullmatch(r"[RMVSI][rmvsi]*", p_) and p_ not in ("Vex", "Evex")]
                if parts and not (d["is4"] and len(role) == 4) and cat["category"] != "GP":
                    cand = parts[min(fx["idx"], len(parts) - 1)]
                    if len(cand.replace("i", "")) == 4: cand = cand[:2] + cand[2:].replace("r", "s")
                    if len([c_ for c_ in cand if c_ in "rmvs"]) == len([o for o in raw_ops if not o.startswith("imm")]) and d["modrm"]:
                        role = cand.replace("i", "")
            # role sanity: AsmJit's JSON roles are sometimes stale; infer from operand count.
            nreg = sum(1 for o in raw_ops if o not in ("imm8", "imm4") and not o.startswith("imm"))
            letters = [c for c in role if c in "rmvs"] if role != "op" else None
            if letters is not None and len(letters) != nreg and d["modrm"]:
                if d["digit"] is not None: role = {1: "m", 2: "vm"}.get(nreg, role)
                elif nreg == 2: role = "mr" if role.startswith("m") else "rm"
                elif nreg == 3: role = role if role in ("mvr", "rmv") else "rvm"
                elif nreg == 4: role = role if role in ("rvms", "rvsm") else "rvms"
            mask = 0
            if any("k" in m.split("|") for m in mods): mask = 1
            if any("z" in m.split("|") for m in mods): mask = 2
            er = "er" in mods; sae = "sae" in mods or er
            lis = d["lvar"] if d["lvar"] is not None else [d["l"] if d["l"] is not None else 0]
            for li in lis:
                alts = []
                try:
                    for o in raw_ops:
                        opts = []
                        for t in o.split("/"):
                            if t in ("b16", "b32", "b64"):
                                opts = [(k, sz, int(t[1:]) // 8) if k == "mem" else (k, sz, 0) for (k, sz, _) in opts]
                                continue
                            t = t.lstrip("~")
                            if t in ("ry", "rv"):
                                opts += [("r32", 0, 0), ("r64", 0, 0)]; continue
                            if t in ("my", "mv"):
                                opts += [("mem", -4, 0), ("mem", -8, 0)]; continue
                            k, sz = expand_tok(t, li if d["lvar"] is not None else (d["l"] or 0))
                            opts.append((k, sz, 0))
                        alts.append([o_ + (any(k_ in ("r32", "r64") for k_, _, _ in opts),) for o_ in opts])
                except KeyError as ex:
                    skipped["tok " + str(ex)] += 1; continue
                # cartesian product of alternatives
                combos = [[]]
                for a in alts: combos = [c + [x] for c in combos for x in a]
                for c in combos:
                    w = d["w"]
                    lowprio = any(k == "mem" and g for k, _, _, g in c)
                    # `sldt r16/m16` (66) vs `sldt r32/m16`: memory takes the form without 66.
                    if cat["category"] == "GP" and gsz is None and 0x66 in d["pre"] and any(k == "mem" for k, _, _, _ in c) and any("r16/m16" in o for o in raw_ops): continue
                    gpy = any(t in ("ry", "rv", "my", "mv") for o in raw_ops for t in o.split("/"))
                    wide = gpy and any(k == "r64" or sz == -8 for k, sz, _, _ in c)
                    ops = [(k, abs(sz), b) for k, sz, b, _ in c]
                    rexw = d["rexw"]
                    if w == 3: w = 1 if wide else 0  # Wy: follows the GP operand size
                    if d["enc"] == "legacy" and wide: rexw = True
                    if d["only"] == "mem" and any(k in VEC + ["mm", "k", "r32", "r64", "tmm"] and i == role.find("m") for i, (k, _, _) in enumerate(ops)): continue
                    if d["only"] == "reg" and any(k == "mem" for k, _, _ in ops): continue
                    # roles
                    r = {"r": None, "m": None, "v": None, "s": None}
                    if role != "op":
                        for i, ch in enumerate(role):
                            if ch in r and i < len(ops): r[ch] = i
                    if r["r"] is not None and ops[r["r"]][0] in ("mem", "vx", "vy", "vz") and r["m"] is not None and ops[r["m"]][0] != "mem":
                        r["r"], r["m"] = r["m"], r["r"]
                    if d["digit"] is not None and r["r"] is not None and r["m"] is None:
                        r["m"], r["r"] = r["r"], None
                    plus = None
                    if d["plus"]:
                        plus = next(i for i, (k, _, _) in enumerate(ops) if k in ("sti", "r8", "r16", "r32", "r64", "xmm", "mm", "k"))
                    if role == "op" and d["modrm"]:
                        mi = [i for i, (k, _, _) in enumerate(ops) if k != "imm"]
                        if mi: r["m"] = mi[0]
                    used = {x for x in r.values() if x is not None} | ({plus} if plus is not None else set())
                    # n for disp8*N: memory operand size; broadcast uses element size at runtime
                    n = 1
                    if d["enc"] == "evex" and fx and fx["n"] != "_":
                        sh = int(fx["n"]); L = li if d["lvar"] is not None else (d["l"] or 0)
                        tt = fx["tt"]
                        if tt in ("FV", "HV", "QV", "FVM", "HVM", "QVM", "OVM"): sh += L
                        elif tt == "T1W": sh += L + (1 if w == 1 else 0)
                        elif tt == "DUP": sh += [0, 2, 3][L]
                        n = 1 << sh
                    elif d["enc"] == "evex":
                        for k, sz, b in ops:
                            if k == "mem" and sz > 0: n = sz
                            if k in ("vx", "vy", "vz"): n = 8 if w == 1 else 4
                        if ins.get("tt") in ("t1s", "t1f") and any(k in ("vx","vy","vz") for k,_,_ in ops): n = 8 if w == 1 else 4
                    form = dict(ops=ops, enc=d["enc"], pre=d["pre"], rexw=rexw, map=d["map"], op=d["op"],
                                digit=d["digit"], imm=d["imm"], pp=d["pp"], w=w, l=li if d["lvar"] is not None else (d["l"] or 0),
                                reg=r["r"], rm=r["m"], vvvv=r["v"], is4=r["s"] if d["is4"] else None, plus=plus,
                                mask=mask, er=er, sae=sae, n=n, sib=d.get("sib", False),
                                evex_mem_bcst=any(b for _, _, b in ops), lowprio=lowprio or cat["category"].startswith("AVX10"),
                                late_vex=d["enc"] == "vex" and name in cpp and cpp[name][2])
                    table.setdefault(name, []).append(form)
    # Unsized far pointer (`lcall ptr(...)`): AsmJit uses the m16:64 form.
    for n_ in ("lcall", "ljmp"):
        if n_ in table: table[n_].sort(key=lambda f: -max([sz for k, sz, _ in f["ops"] if k == "mem"] or [0]))
    # EVEX scalar stores (vmovss/vmovsd/vmovsh m, xmm) accept {k} in AsmJit.
    for n_, forms in table.items():
        if any(f["enc"] == "evex" and f["mask"] for f in forms):
            for f in forms:
                if f["enc"] == "evex" and not f["mask"] and f["ops"] and f["ops"][0][0] == "mem": f["mask"] = 1
    KMAP = {"xmm": "xmm", "ymm": "ymm", "zmm": "zmm", "mm": "mm", "k": "k", "tmm": "tmm", "st0": "st0", "sti": "sti",
            "r8": "r8", "r16": "r16", "r32": "r32", "r64": "r64", "ax": "ax", "al": "al", "cl": "cl", "eax": "eax", "rax": "rax", "dx": "dx", "mem": "mem", "vx": "vx", "vy": "vy", "vz": "vz", "imm": "imm", "maddr": "maddr"}
    def opt(v): return "null" if v is None else str(v)
    lines = ["// Generated by scripts/gen_vec_tables.py from ref/asmjit/db/isa_x86.json. Do not edit.",
             "const Form = @import(\"vec.zig\").Form;", "", "pub const Mnemonic = enum {"]
    lines += [f"    @\"{n}\"," for n in table]
    lines += ["};", "", "pub const table = struct {"]
    nforms = 0
    for n, forms in table.items():
        lines.append(f"    pub const @\"{n}\" = [_]Form{{")
        for f in forms:
            nforms += 1
            ops = ", ".join(f".{{ .k = .{KMAP[k]}, .size = {sz}, .bcst = {b} }}" for k, sz, b in f["ops"])
            pre = ", ".join(f"0x{x:02x}" for x in f["pre"]); op = ", ".join(f"0x{x:02x}" for x in f["op"])
            lines.append(f"        .{{ .ops = &.{{{ops}}}, .enc = .{f['enc']}, .pre = &.{{{pre}}}, .rexw = {str(f['rexw']).lower()}, "
                         f".map = {f['map']}, .op = &.{{{op}}}, .digit = {opt(f['digit'])}, .imm = {f['imm']}, .pp = {f['pp']}, "
                         f".w = {f['w']}, .l = {f['l']}, .reg = {opt(f['reg'])}, .rm = {opt(f['rm'])}, .vvvv = {opt(f['vvvv'])}, "
                         f".is4 = {opt(f['is4'])}, .plus = {opt(f['plus'])}, .mask = {f['mask']}, .er = {str(f['er']).lower()}, "
                         f".sae = {str(f['sae']).lower()}, .n = {f['n']}, .lowprio = {str(f['lowprio']).lower()}, .late_vex = {str(f['late_vex']).lower()} }},")
        lines.append("    };")
    lines.append("};")
    open(os.path.join(ROOT, "src/x86_64/vec_table.zig"), "w").write("\n".join(lines) + "\n")
    print(f"{nforms} forms, {len(table)} mnemonics; skipped: {dict(skipped)}")

if __name__ == "__main__":
    main()
