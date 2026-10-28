#!/usr/bin/env python3
"""Generate src/aarch64/a64_table.zig from AsmJit's AArch64 ISA database (db/isa_aarch64.json).

Each DB form ("add Xd|SP, Xn|SP, #immZ, {lsl #n=0|12}" + bit pattern "10010001|0|n:1|immZ:12|Rn|Rd")
becomes a Zig `Form`: fixed bits, operand specs (what each operand must be and which named value it
provides) and field placements (where each named value goes). Immediate transforms named by the
DB `imm` key (LogicalImm, ImmWide, ASimdShiftNImm, ...) are implemented in src/aarch64/a64.zig.
Covers the GP and ASIMD categories (incl. crypto, PAUTH, MTE); SVE/SME are skipped.

    python3 scripts/gen_a64_tables.py
"""
import json, os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "ref/asmjit/db/isa_aarch64.json")
WIDTH = {"Rd": 5, "Rn": 5, "Rm": 5, "Rs": 5, "Rt": 5, "Ra": 5, "Rx": 5, "Rd2": 5, "Rs2": 5, "Rt2": 5,
         "sz": 2, "cond": 4, "nzcv": 4, "W": 1, "!post": 1, "CRm": 4, "CRn": 4, "sop": 2, "option": 3}
ARRS = ["8B", "16B", "4H", "8H", "2S", "4S", "1D", "2D", "1Q", "4B", "2H", "2B", "1S", "2D"]
ARR_Z = {a: "a" + a.lower() for a in ARRS}  # Zig enum tag: a8b, a16b, ...

names = set()  # every named value (fields + operand values)

class Skip(Exception):
    pass

def parse_bits(op):
    """-> (fixed bits, [(name, at, width, src)]) ; repeated plain pieces are MSB-first."""
    pieces = []
    for p in op.split("|"):
        p = p.replace(" ", "")
        if not p: raise Skip("empty piece")
        if p == "W1": pieces += [("W", 1, None), ("1", 1, None)]; continue
        if re.fullmatch(r"[01xX]+", p): pieces.append((p, len(p), None)); continue
        m = re.fullmatch(r"(!?[A-Za-z_][A-Za-z_0-9]*)\[(\d+)(?::(\d+))?\]", p)
        if m:
            hi = int(m.group(2)); lo = int(m.group(3)) if m.group(3) else hi
            pieces.append((m.group(1), hi - lo + 1, lo)); continue
        m = re.fullmatch(r"(!?[A-Za-z_][A-Za-z_0-9]*)(?::(\d+))?", p)
        if not m: raise Skip("piece " + p)
        n = m.group(1)
        w = int(m.group(2)) if m.group(2) else WIDTH.get(n)
        if w is None: raise Skip("width " + n)
        pieces.append((n, w, None))
    if sum(w for _, w, _ in pieces) != 32: raise Skip("bits != 32")
    # src offsets for plain repeated names: first occurrence is most significant
    tot = collections.Counter(); 
    for n, w, s in pieces:
        if s is None and not re.fullmatch(r"[01xX]+", n) and not (n.startswith("R") and w == 5): tot[n] += w
    used = collections.Counter()
    fixed, fields, at = 0, [], 32
    for n, w, s in pieces:
        at -= w
        if re.fullmatch(r"[01xX]+", n):
            fixed |= int(n.replace("x", "0").replace("X", "0"), 2) << at
            continue
        if s is None and n.startswith("R") and w == 5:
            s = 0  # repeated register field (cinc: Rn twice) = same value twice
        elif s is None:
            used[n] += w; s = tot[n] - used[n]
        fields.append((n, at, w, s)); names.add(n)
    return fixed, fields

def split_ops(s):
    out, dep, c = [], 0, ""
    for ch in s:
        if ch in "[{": dep += 1
        if ch in "]}": dep -= 1
        if ch == "," and dep == 0: out.append(c.strip()); c = ""
        else: c += ch
    if c.strip(): out.append(c.strip())
    return out

def regname(suffix):
    return "R" + suffix

def op_spec(t, form):
    """Return Zig OpSpec literal fields as a dict."""
    # GP registers
    m = re.fullmatch(r"([WX])(d|n|m|s|t|a|x|dn)(2?)(\|W?SP)?", t)
    if m:
        f = regname(m.group(2) + m.group(3)); names.add(f)
        k = ("wsp" if m.group(1) == "W" else "xsp") if m.group(4) else m.group(1).lower()
        return dict(k=k, f=f)
    if t in ("Rm", "Rn", "Rd"):
        names.add(t); return dict(k="rext", f=t)
    # scalar FP/SIMD registers
    m = re.fullmatch(r"([BHSDQ])(d|n|m|s|t|a|x)(2?)", t)
    if m:
        f = regname(m.group(2) + m.group(3)); names.add(f)
        return dict(k="s" + m.group(1).lower(), f=f)
    # vector with arrangement
    m = re.fullmatch(r"[VD]([dnmsxa])\.(t|ta|tb|\d+[BHSDQ])", t)
    if m:
        f = regname(m.group(1)); names.add(f)
        a = m.group(2)
        if a == "t": return dict(k="vt", f=f)
        if a == "ta": return dict(k="vta", f=f)
        if a == "tb": return dict(k="vtb", f=f)
        return dict(k="vfix", f=f, arr=ARR_Z[a])
    # vector element
    m = re.fullmatch(r"V([dnmsx])\.(\d*)([BHSD])\[#(idx\d?|\d+)\]", t)
    if m:
        f = regname(m.group(1)); names.add(f)
        d = dict(k="velem", f=f, et=m.group(3).lower())
        if m.group(2): d["arr"] = ARR_Z[m.group(2) + m.group(3)]
        if m.group(4).isdigit(): d["fixed"] = int(m.group(4))
        else: d["f2"] = m.group(4); names.add(m.group(4))
        return d
    # register lists
    m = re.fullmatch(r"(\d)x\{V([dnsx])\.(t|\d+[BHSD]|[BHSD])\}(\+?)(\[#idx\])?", t)
    if m:
        f = regname(m.group(2)); names.add(f)
        d = dict(k="vlist", f=f, n=int(m.group(1)))
        a = m.group(3)
        if m.group(5):
            d["k"] = "vlistelem"; d["et"] = a.lower(); d["f2"] = "idx"; names.add("idx")
        elif a != "t": d["arr"] = ARR_Z[a]
        return d
    m = re.fullmatch(r"(\d)x\{([WX])([dst])\}\+", t)
    if m:
        f = regname(m.group(3)); names.add(f)
        return dict(k="glist", f=f, n=int(m.group(1)), w=m.group(2) == "X")
    # immediates
    if t in ("#0", "#8", "#16", "#32"): return dict(k="immfix", fixed=int(t[1:]))
    m = re.fullmatch(r"#relS(\*4)?", t)
    if m: names.add("relS"); return dict(k="rel", f="relS", scale=2 if m.group(1) else 0)
    if t == "#cond": names.add("cond"); return dict(k="cond", f="cond")
    if t == "#fimm": names.add("fimm"); return dict(k="fimm", f="fimm")
    m = re.fullmatch(r"\{?#(\w+)(?:=(\d+))?\}?", t)
    if m:
        n = m.group(1); names.add(n)
        d = dict(k="imm", f=n)
        if t.startswith("{"): d["opt"] = True; d["fixed"] = int(m.group(2) or 0)
        return d
    # shifts / extends (optional trailing operand)
    m = re.fullmatch(r"\{(lsl\|lsr\|asr|sop|lsl|extend|msl|lsl\|msl) #n(?:=(\d+)\|(\d+))?\}", t)
    if m:
        names.add("n")
        allowed = {"lsl|lsr|asr": "shift3", "sop": "shift4", "lsl": "lsl", "extend": "extend", "msl": "msl", "lsl|msl": "lslmsl"}[m.group(1)]
        d = dict(k="shift", sh=allowed, f="n", opt=True)
        if m.group(2): d["scale"] = int(m.group(3))  # n field = amount / 12
        return d
    # memory
    m = re.fullmatch(r"\[(Xn\|SP|PC)(?:, (.*))?\](\{@\}\{!\}|\{!\}|@|!)?", t)
    if m:
        d = dict(k="mem", mode={None: "off", "@": "post", "!": "pre", "{@}{!}": "any", "{!}": "offpre"}[m.group(3)])
        if m.group(1) == "PC": d["k"] = "memlit"
        else: names.add("Rn")
        inner = m.group(2)
        if inner is None: d["mk"] = "base"; return d
        mm = re.fullmatch(r"#(off[SZ])(?:\*(\d+))?", inner)
        if mm:
            names.add(mm.group(1)); d.update(mk="imm", f=mm.group(1), scale=int(mm.group(2) or 1).bit_length() - 1, signed=mm.group(1) == "offS")
            return d
        mm = re.fullmatch(r"#off==?(\d+)(<<sz)?", inner) or re.fullmatch(r"#(\d+)", inner)
        if mm:
            d.update(mk="fixed", fixed=int(mm.group(1)))
            if len(mm.groups()) > 1 and mm.group(2): d["mk"] = "fixedsz"
            return d
        if inner == "Xm": names.add("Rm"); d.update(mk="postreg"); return d
        mm = re.fullmatch(r"Rm, \{uxtw\|lsl\|sxtw\|sxtx #n(?:\*(\d))?\}", inner)
        if mm:
            names.update(["Rm", "option", "n"]); d.update(mk="reg", scale=int(mm.group(1)) if mm.group(1) else -1)
            return d
        raise Skip("mem " + inner)
    raise Skip("op " + t)

def zarr(lst):
    return "&.{" + ", ".join("." + (ARR_Z[a] if a != "~" else "none") for a in lst) + "}"

OVR = os.path.join(ROOT, "scripts/a64_overrides.json")

def main(write=True):
    db = json.load(open(DB))
    # Fixed-bit corrections (stale U/opc bits in the JSON DB), as XOR masks keyed by "inst | op".
    # Derived from AsmJit's expected encodings by scripts/derive_a64_overrides.py.
    ovr = json.load(open(OVR)) if os.path.exists(OVR) else {}
    # Forms missing from the DB: `ret` without an operand (= ret x30).
    db["instructions"].append({"category": "BASE", "data": [{"inst": "ret", "op": "11010110|010|11111|0|00000|11110|00000"}]})
    table = collections.OrderedDict()
    skipped = collections.Counter()
    for g in db["instructions"]:
        cat = g["category"]
        if cat.startswith("SVE") or cat == "SME": continue
        for i in g["data"]:
            inst = i["inst"]
            mns, _, opstr = inst.partition(" ")
            mns = mns.split("|")
            try:
                fixed, fields = parse_bits(i["op"])
                fixed ^= int(ovr.get(inst + " | " + i["op"], "0"), 16)
                ops = [op_spec(t, i) for t in split_ops(opstr)]
            except Skip as e:
                skipped[str(e).split(" ")[0]] += 1
                continue
            t = ta = tb = None
            if "t" in i:
                v = i["t"].split()
                if "." in v[-1]: ta = [x.split(".")[0] if x != "~" else "~" for x in v]; tb = [x.split(".")[1] if x != "~" else "~" for x in v]
                else: t = v
            if "ta.tb" in i:
                v = i["ta.tb"].split()
                ta = [x.split(".")[0] if x != "~" else "~" for x in v]; tb = [x.split(".")[1] if x != "~" else "~" for x in v]
            if "ta" in i: ta = i["ta"].split()
            if "tb" in i: tb = i["tb"].split()
            # Q (bit 30) decides 64- vs 128-bit arrangements; the DB lists are sometimes wrong.
            q_fixed = not any(at <= 30 < at + w for _, at, w, _ in fields)
            if q_fixed:
                q = (fixed >> 30) & 1
                M64 = {"16B": "8B", "8H": "4H", "4S": "2S", "2D": "1D"}
                M128 = {v: k for k, v in M64.items()}
                conv = lambda lst, m: None if lst is None else [m.get(a, a) for a in lst]
                if t and any(o.get("k") in ("vt", "vlist") for o in ops):
                    t = conv(t, M64 if q == 0 else M128)
            immfn, immargs = "none", []
            if "imm" in i:
                mm = re.fullmatch(r"(\w+)\((.*)\)", i["imm"])
                immfn = mm.group(1); immargs = [a.strip() for a in mm.group(2).split(",")]
            # Pre/post-index and unscaled (imm9) offsets are byte offsets; the DB's "*8" only
            # holds for the MTE tag instructions (scaled by 16).
            for o in ops:
                if o.get("mk") == "imm" and o.get("f") == "offS" and any(n_ == "offS" and w_ == 9 for n_, _, w_, _ in fields) \
                        and not re.match(r"(st|stz|st2|stz2|ld)g\b", mns[0]):
                    o["scale"] = 0
            for mn in mns:
                if mn.endswith(".<cond>"): ops = [dict(k="cond", f="cond")] + ops
                mn = mn.replace(".<cond>", "_cond")
                table.setdefault(mn, []).append(dict(fixed=fixed, fields=fields, ops=ops, t=t, ta=ta, tb=tb, immfn=immfn, immargs=immargs, inst=inst, key=inst + " | " + i["op"]))
    if not write: return table
    fns = sorted({f["immfn"] for v in table.values() for f in v})
    lines = ["// Generated by scripts/gen_a64_tables.py from ref/asmjit/db/isa_aarch64.json. Do not edit.",
             'const a64 = @import("a64.zig");', "const Form = a64.Form;", "",
             "pub const Name = enum(u8) {"] + [f'    @"{n}",' for n in sorted(names)] + ["};", "",
             "pub const ImmFn = enum {"] + [f"    {n}," for n in fns] + ["};", "",
             "pub const Mnemonic = enum {"] + [f'    @"{m}",' for m in table] + ["};", "", "pub const table = struct {"]
    nf = 0
    def lit(d):
        parts = []
        for k_, v_ in d.items():
            if k_ in ("k", "f", "f2", "arr", "et", "sh", "mode", "mk"):
                parts.append(f".{k_} = ." + (f'@"{v_}"' if k_ in ("f", "f2") else v_))
            elif isinstance(v_, bool): parts.append(f".{k_} = {str(v_).lower()}")
            else: parts.append(f".{k_} = {v_}")
        return ".{ " + ", ".join(parts) + " }"
    for mn, forms in table.items():
        lines.append(f'    pub const @"{mn}" = [_]Form{{')
        for f in forms:
            nf += 1
            flds = ", ".join(f'.{{ .n = .@"{n}", .at = {a}, .w = {w}, .src = {s} }}' for n, a, w, s in f["fields"])
            ops = ", ".join(lit(o) for o in f["ops"])
            extra = ""
            if f["t"]: extra += f", .t = {zarr(f['t'])}"
            if f["ta"]: extra += f", .ta = {zarr(f['ta'])}"
            if f["tb"]: extra += f", .tb = {zarr(f['tb'])}"
            if f["immfn"] != "none":
                args = []
                for a in f["immargs"]:
                    if re.fullmatch(r"-?\d+", a): args.append(a)
                    elif a == "x": args.append("-1")
                    elif a == "sz": args.append("-2")
                    else: args.append("-3")
                extra += f", .imm = .{f['immfn']}, .args = &.{{ {', '.join(args)} }}"
            if mn in ("cset", "csetm", "cinc", "cinv", "cneg"): extra += ", .inv_cond = true"
            lines.append(f"        // {f['inst']}")
            lines.append(f"        .{{ .bits = 0x{f['fixed']:08X}, .fields = &.{{{flds}}}, .ops = &.{{{ops}}}{extra} }},")
        lines.append("    };")
    lines.append("};")
    open(os.path.join(ROOT, "src/aarch64/a64_table.zig"), "w").write("\n".join(lines) + "\n")
    print(f"{nf} forms, {len(table)} mnemonics; skipped: {dict(skipped)}")

if __name__ == "__main__":
    main()
