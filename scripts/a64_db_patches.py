"""Corrections to AsmJit's isa_aarch64.json (wrong operand names/arrangements, missing forms),
applied by gen_a64_tables.py before parsing. Each fix was checked against AsmJit's expected
encodings (asmjit_test_assembler_a64.cpp) and the Arm ARM."""
import re

def _sub(i, inst=None, op=None, **kw):
    if inst: i["inst"] = re.sub(inst[0], inst[1], i["inst"])
    if op: i["op"] = re.sub(op[0], op[1], i["op"])
    i.update(kw)

def patch(db):
    extra = []
    for g in db["instructions"]:
        if g["category"].startswith("SVE") or g["category"] == "SME": continue
        for i in g["data"]:
            inst, op = i["inst"], i["op"]
            mn = inst.split(" ")[0]
            if mn in ("crc32x", "crc32cx"): _sub(i, inst=(r"Xd, Xn", "Wd, Wn"))
            elif mn == "frecpx": _sub(i, inst=(r", [SDH]m$", ""))
            elif mn == "stxrh": _sub(i, inst=(r"Wd, Xs", "Ws, Wd"), op=(r"\|Rd\|0\|11111\|Rn\|Rs$", "|Rs|0|11111|Rn|Rd"))
            elif re.fullmatch(r"ldset(a|l|al)?", mn) and inst.startswith(mn + " Xs, Wd"): _sub(i, inst=(r"Xs, Wd", "Xs, Xd"))
            elif mn == "xpaclri": i["inst"] = "xpaclri"
            elif mn == "mvn" and op.startswith("X0101010"):
                w = "1" if " Xd" in inst else "0"
                _sub(i, inst=(r"Wn|Xn", lambda m: m.group(0)[0] + "m"), op=(r"^X0101010\|sop\|1\|Rn\|n:6\|Rn\|Rd$", w + "0101010|sop:2|1|Rm|n:6|11111|Rd"))
            elif mn in ("trn1", "trn2") and i.get("t") == "16B 8H 4S": i["t"] = "16B 8H 4S 2D"
            elif mn in ("fcvtn", "fcvtn2") and re.search(r"Vd\.\d[SD], Vn\.\d[HS]$", inst):
                a, b = re.search(r"Vd\.(\w+), Vn\.(\w+)$", inst).groups()
                fix = {("4S", "4H"): ("4H", "4S"), ("4S", "8H"): ("8H", "4S"), ("2D", "2S"): ("2S", "2D"), ("2D", "4S"): ("4S", "2D")}[(a, b)]
                i["inst"] = f"{mn} {'Vx' if mn.endswith('2') else 'Vd'}.{fix[0]}, Vn.{fix[1]}"
                if mn.endswith("2"): _sub(i, op=(r"\|Rd$", "|Rx"))
            elif mn == "usdot" and "Vm.tb" in inst: _sub(i, inst=(r"Vm\.tb", "Vm.8B" if "2S" in inst else "Vm.16B"))
            elif mn in ("sqshrn", "sqshrn2", "uqshrn", "uqshrn2", "sqrshrn", "sqrshrn2", "uqrshrn", "uqrshrn2", "sqshrun", "sqshrun2", "sqrshrun", "sqrshrun2") and "ta.tb" in i:
                i["imm"] = re.sub(r",\s*\d+\)$", ", sz)", i.get("imm", ""))
            elif mn in ("sshll", "sshll2", "ushll", "ushll2"): _sub(i, op=(r"\|Rx$", "|Rd"))
            elif mn in ("lsl",) and "#n" in inst: i["imm"] = "ImmLSL(n, " + ("1" if " Xd" in inst else "0") + ")"
            elif mn in ("lsr", "asr") and "#n" in inst: _sub(i, inst=(r"#n$", "#immr"))
            elif mn in ("bfi", "bfc"): i["imm"] = "ImmBFIZ(lsb, width, " + ("1" if " Xd" in inst else "0") + ")"
            elif mn == "bfxil": i["imm"] = "ImmBFX(lsb, width, " + ("1" if " Xd" in inst else "0") + ")"
            elif mn == "adrp": _sub(i, inst=(r"#relS$", "#relS*4096"))
            elif mn in ("ldraa", "ldrab"): _sub(i, inst=(r"#offS\]", "#offS*8]"))
            elif mn in ("addg", "subg"): _sub(i, inst=(r"#imm1,", "#imm1*16,"))
            elif mn == "prfm" and "#offZ]" in inst: _sub(i, inst=(r"#offZ\]", "#offZ*8]"))
            elif mn in ("ld3", "st3"): _sub(i, inst=(r"#off==16\]", "#off==24]")); _sub(i, inst=(r"#off==32\]", "#off==48]"))
            elif re.fullmatch(r"fml[as]l2?", mn): _sub(i, op=(r"imm\[", "idx["))
            elif mn in ("and", "ands", "bic", "bics", "orr", "orn", "eor", "eon") and re.search(r"[WX]d\|W?SP, [WX]n, [WX]m", inst):
                _sub(i, inst=(r"([WX]d)\|W?SP", r"\1"))
            elif mn == "ands" and "#imm" in inst: _sub(i, inst=(r"([WX]d)\|W?SP", r"\1"))
            elif mn == "clrex": i["imm"] = "ImmISB(imm)"
            elif mn == "sys": _sub(i, inst=(r"#Cn, #Cm", "#CRn, #CRm"))
            elif mn in ("movi", "mvni"): _sub(i, inst=(r"\{lsl #n\}", "{lsl|msl #n}"))
            elif mn == "dsb" and "barrier_op" in inst: _sub(i, op=(r"\|CRm\|101\|", "|CRm|100|"))
            elif mn == "chkfeat": i["inst"] = "chkfeat X16"
            elif mn in ("mrs", "msr") and "#sysreg" in inst: i["imm"] = "ImmSysReg(sysreg)"
            elif mn.startswith("sm3tt"): _sub(i, inst=(r"Vm\.4\[#idx\]", "Vm.S[#idx]"))
        if g["category"] == "ASIMD" or any(d["inst"].startswith("fcvtn ") for d in g["data"]):
            pass
    for g in db["instructions"]:
        if not g["category"].startswith("SVE"): continue
        for i in g["data"]:
            i["inst"] = i["inst"].replace("#idxmm", "#imm")
            if i["inst"].startswith("while") and "|000X" in i["op"]:
                i["op"] = i["op"].replace("|000X", "|000|sf:1|")  # sf: X (1) or W (0) operands
            # Prefer DUP (imm8) over DUPM for `mov Zd.t, #imm` (as LLVM and AsmJit do).
        mv = [x for x in g["data"] if x["inst"] == "mov Zd.t, #imm, {lsl #n}"]
        if mv:
            g["data"] = [x for x in g["data"] if x not in mv]
            k = next(j for j, x in enumerate(g["data"]) if x["inst"] == "mov Zd.t, #imm")
            g["data"][k:k] = mv
    # CMN/CMP: prefer the shifted-register form over the extended one (as AsmJit does) unless SP is used.
    for g in db["instructions"]:
        d = g["data"]
        ext = [x for x in d if re.match(r"(cmn|cmp) [WX]n\|W?SP, (Wm|Rm), \{extend", x["inst"])]
        if ext:
            rest = [x for x in d if x not in ext]
            g["data"] = rest + ext
    # Missing forms.
    extra += [
        {"inst": "ret", "op": "11010110|010|11111|0|00000|11110|00000"},
        {"inst": "fcvtxn Sd, Dn", "op": "01111110|01|10000|10110|10|Rn|Rd"},
        {"inst": "fcvtxn Vd.2S, Vn.2D", "op": "00101110|01|10000|10110|10|Rn|Rd"},
        {"inst": "fcvtxn2 Vx.4S, Vn.2D", "op": "01101110|01|10000|10110|10|Rn|Rx"},
    ]
    db["instructions"].append({"category": "BASE", "data": extra})
