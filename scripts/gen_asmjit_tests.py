#!/usr/bin/env python3
"""Port AsmJit's x86-64 assembler tests (asmjit_test_assembler_x64.cpp) to ZigJIT.

Every TEST_INSTRUCTION("hex", inst(ops...)) is translated to a `gp.emit` check when its
mnemonic and operands are expressible in ZigJIT; the rest are counted as unsupported with a
reason. Output: src/asmjit_x64_tests.zig + bench/asmjit_x64_unsupported.txt (summary).

    python3 scripts/gen_asmjit_tests.py [path/to/asmjit]   # default ../asmjit-loc
"""
import re, sys, collections, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
ref = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root.parent / "asmjit-loc")
src = (ref / "asmjit-testing/tests/asmjit_test_assembler_x64.cpp").read_text()
mnems = set(re.findall(r"^    (\w+|@\"\w+\"),$", (root / "src/x86_64/gp_table.zig").read_text().split("pub const Mnemonic = enum {")[1].split("};")[0], re.M))
mnems = {m.strip('@"') for m in mnems}
vmnems = set(re.findall(r'^    @"(\w+)",$', (root / "src/x86_64/vec_table.zig").read_text().split("pub const Mnemonic = enum {")[1].split("};")[0], re.M))
VREG = {}
for cls, n in (("xmm", 32), ("ymm", 32), ("zmm", 32), ("mm", 8), ("k", 8), ("tmm", 8)):
    for i in range(n): VREG[f"{cls}{i}"] = f"vec.{cls}({i})"
for i in range(8): VREG[f"st{i}"] = f"vec.st({i})"
for i, sg in enumerate(("es", "cs", "ss", "ds", "fs", "gs")): VREG[sg] = f"vec.seg({i})"
for i in range(16): VREG[f"cr{i}"] = f"vec.cr({i})"; VREG[f"dr{i}"] = f"vec.dr({i})"
VSZ = {"byte_ptr": 1, "word_ptr": 2, "dword_ptr": 4, "qword_ptr": 8, "tbyte_ptr": 10, "fword_ptr": 6,
       "xmmword_ptr": 16, "ymmword_ptr": 32, "zmmword_ptr": 64, "ptr": 0}
RC = {"rn_sae": ".rn", "rd_sae": ".rd", "ru_sae": ".ru", "rz_sae": ".rz"}

# AsmJit condition-code aliases -> the spelling used by ZigJIT's tables.
CC = {"a": "nbe", "ae": "nb", "c": "b", "e": "z", "g": "nle", "ge": "nl", "na": "be", "nae": "b",
      "nc": "nb", "ne": "nz", "ng": "le", "nge": "l", "pe": "p", "po": "np"}
def alias(mn):
    if mn == "sal": return "shl"
    for p in ("cmov", "set", "j"):
        if mn.startswith(p) and mn[len(p):] in CC: return p + CC[mn[len(p):]]
    return mn

def vmem(fn, args, bc):
    p = mem_parts(args); sz = VSZ[fn]
    f = []
    if p["base"]: f.append(f".base = .{p['base']}")
    if p["index"]: f.append(f".index = .{p['index']}")
    if p["vindex"]: f.append(f".vindex = {p['vindex']}")
    f += [f".scale = .x{1 << p['shift']}", f".disp = {p['disp']}", f".size = {sz}", f".bcst = {bc}"]
    if p["rip"]: f.append(".rip = true")
    return "vec.Mem{ " + ", ".join(f) + " }"

def voperand(t):
    t = t.strip()
    if t in VREG: return VREG[t]
    if t in REG:
        s_, g = REG[t]; return f"gp.{RFN[s_]}(.{g})"
    if t in HI8: return f"gp.r8hi(.r{t[0]}x)"
    bc = 0
    mb = re.fullmatch(r"(.*)\._1to(\d+)\(\)", t)
    if mb: t, bc = mb.group(1), int(mb.group(2))
    m = re.fullmatch(r"(\w+)\((.*)\)", t)
    if m and m.group(1) in VSZ: return vmem(m.group(1), m.group(2), bc)
    if m: raise Unsup("operand:" + m.group(1))
    if re.fullmatch(r"[a-z]\w*", t): raise Unsup("reg-class:" + re.sub(r"\d+$", "", t))
    return str(imm(t))

GP = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"] + [f"r{i}" for i in range(8, 16)]
REG = {}
for i, g in enumerate(GP):
    if i < 8:
        b = ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil"][i]
        w = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"][i]
        REG[b] = ("b", g); REG[w] = ("w", g); REG["e" + w] = ("d", g); REG[g] = ("q", g)
    else:
        REG[g + "b"] = ("b", g); REG[g + "w"] = ("w", g); REG[g + "d"] = ("d", g); REG[g] = ("q", g)
HI8 = {"ah", "ch", "dh", "bh"}
SZ = {"byte_ptr": "b", "word_ptr": "w", "dword_ptr": "d", "qword_ptr": "q", "ptr": None}
FN = {"b": "byte", "w": "word", "d": "dword", "q": "qword", None: "mem"}
RFN = {"b": "r8", "w": "r16", "d": "r32", "q": "r64"}

class Unsup(Exception):
    pass

def split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(": depth += 1
        if ch == ")": depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def imm(t):
    t = t.strip()
    m = re.fullmatch(r"(-?)(0x[0-9a-fA-F]+|\d+)(u|ull|ll|LL|ULL|U)?", t)
    if not m: raise Unsup("operand:" + t.split("(")[0])
    v = int(m.group(2), 0)
    return -v if m.group(1) else v

def mem_parts(args):
    a = split_args(args)
    p = dict(base=None, index=None, vindex=None, shift=0, disp=0, rip=False, addr32=False)
    if not a: raise Unsup("mem-base")
    if a[0] == "rip": p["rip"] = True
    elif a[0] in REG and REG[a[0]][0] == "q": p["base"] = REG[a[0]][1]
    else:
        try: p["disp"] = imm(a[0])
        except Unsup: raise Unsup("mem-base")
    rest = a[1:]
    if rest and (rest[0] in REG or rest[0] in VREG):
        if rest[0] in VREG: p["vindex"] = VREG[rest[0]]
        elif REG[rest[0]][0] == "q": p["index"] = REG[rest[0]][1]
        elif REG[rest[0]][0] == "d": p["index"] = REG[rest[0]][1]; p["addr32"] = True  # AsmJit encodes it like the 64-bit index
        else: raise Unsup("mem-index")
        rest = rest[1:]
        if rest: p["shift"] = imm(rest[0]); rest = rest[1:]
    if rest: p["disp"] += imm(rest[0])
    if not -2**31 <= p["disp"] < 2**31: raise Unsup("moffs")
    return p

def xmem(p):
    f = []
    if p["base"]: f.append(f".base = .{p['base']}")
    if p["index"]: f.append(f".index = .{p['index']}")
    f.append(f".scale = .x{1 << p['shift']}"); f.append(f".disp = {p['disp']}")
    return "x.Mem{ " + ", ".join(f) + " }"

def mem(fn, args):
    p = mem_parts(args)
    if p["rip"] or p["vindex"] or p["addr32"]: raise Unsup("mem-kind")
    return xmem(p)

def operand(t, regsize):
    t = t.strip()
    if t in REG:
        s, g = REG[t]; return f"gp.{RFN[s]}(.{g})", s
    if t in HI8: return f"gp.r8hi(.r{t[0]}x)", "b"
    ma = re.fullmatch(r"ptr_abs\((0x[0-9a-fA-F]+|\d+)\)", t)
    if ma and int(ma.group(1), 0) < 2**31:
        return f"gp.mem(.{{ .disp = {int(ma.group(1), 0)} }})", None
    m = re.fullmatch(r"(\w+)\((.*)\)", t)
    if m and m.group(1) in SZ and mem_parts(m.group(2))["rip"]:
        p = mem_parts(m.group(2))
        return f"gp.rip({p['disp']}, .{SZ[m.group(1)] or 'none'})", None
    if m and m.group(1) in SZ:
        s = SZ[m.group(1)]
        return f"gp.{FN[s]}({mem(m.group(1), m.group(2))})", None
    if m: raise Unsup("operand:" + m.group(1))
    if re.fullmatch(r"[a-z]\w*", t): raise Unsup("reg-class")
    return str(imm(t)), None

cases, unsup, total = [], collections.Counter(), 0
unsup_cases = []
for hexs, body in re.findall(r'TEST_INSTRUCTION\("([0-9A-F]*)"\s*,\s*(.*)\);', src):
    total += 1
    body = body.strip()
    try:
        extra = []; gopt = []
        while True:
            m = re.match(r"(\w+)\((\w*)\)\.", body)
            if not m: break
            p, arg = m.group(1), m.group(2)
            if p == "k" and arg in VREG: extra.append(f".k = {arg[1:]}")
            elif p == "z": extra.append(".z = true")
            elif p in RC: extra.append(f".rc = {RC[p]}")
            elif p == "sae": extra.append(".sae = true")
            elif p == "evex": extra.append(".evex = true")
            elif p == "vex": extra.append(".vex = true")
            elif p == "lock": gopt.append(".lock = true"); extra.append(".lock = true")
            elif p == "rex": gopt.append(".rex = true"); extra.append(".rex = true")
            elif p == "mod_mr": gopt.append(".mod = .mr")
            elif p == "mod_rm": gopt.append(".mod = .rm")
            elif p == "long_": gopt.append(".long = true")
            else: raise Unsup("prefix:" + p)
            body = body[m.end():]
        m = re.fullmatch(r"(\w+)\((.*)\)", body)
        if not m: raise Unsup("syntax")
        mn = m.group(1).rstrip("_")
        mn = alias(mn)
        ops = split_args(m.group(2))
        txt = (hexs + " " + body).replace("\\", "\\\\").replace('"', '\\"')
        bytes_ = ", ".join(f"0x{hexs[i:i+2]}" for i in range(0, len(hexs), 2))
        allgp = all(o in REG or o in HI8 or not re.search(r"[xyz]mm|\bk\d|\bmm\d|tmm|_1to|st\d|^(es|cs|ss|ds|fs|gs|cr\d+|dr\d+)$", o) and not re.search(r"(xmm|ymm|zmm|tbyte|fword)word_ptr|tbyte_ptr|fword_ptr", o) for o in ops)
        mo = None
        if mn in ("mov", "movabs") and len(ops) == 2:
            for i_, o_ in enumerate(ops):
                mm_ = re.fullmatch(r"(?:ptr_abs|ptr)\((0x[0-9A-Fa-f]+|\d+)\)", o_)
                if mm_ and (mn == "movabs" or int(mm_.group(1), 0) >= 2**31) and ops[1 - i_] in REG:
                    s_, g_ = REG[ops[1 - i_]]
                    mo = f"check3(&.{{ {bytes_} }}, {str(i_ == 0).lower()}, gp.{RFN[s_]}(.{g_}), {int(mm_.group(1), 0)}, \"{txt}\")"
        if mo:
            cases.append(f"    fails += {mo};")
        elif mn in mnems and allgp and not [e_ for e_ in extra if e_ not in (".lock = true", ".rex = true")]:
            zops = [operand(o, None)[0] for o in ops]
            if mn == "xchg" and len(ops) == 2 and all(o in REG or o in HI8 for o in ops) and not gopt: zops.reverse()
            cases.append(f'    fails += check(&.{{ {bytes_} }}, .@"{mn}", .{{ {", ".join(zops)} }}, .{{ {", ".join(gopt)} }}, "{txt}");')
        elif mn in vmnems:
            if any(g_ not in (".lock = true", ".rex = true") for g_ in gopt): raise Unsup("prefix:vec")
            zops = [voperand(o) for o in ops]
            cases.append(f'    fails += checkV(&.{{ {bytes_} }}, .@"{mn}", .{{ {", ".join(zops)} }}, .{{ {", ".join(extra)} }}, "{txt}");')
        elif mn in mnems and not extra:
            zops = [operand(o, None)[0] for o in ops]
            cases.append(f'    fails += check(&.{{ {bytes_} }}, .@"{mn}", .{{ {", ".join(zops)} }}, .{{ {", ".join(gopt)} }}, "{txt}");')
        else:
            raise Unsup("mnemonic:" + mn)
    except Unsup as e:
        unsup[str(e).split(":")[0]] += 1
        unsup["  " + str(e)] += 1
        unsup_cases.append(f"{e}: {hexs} {body}")

out = [f"//! GENERATED by scripts/gen_asmjit_tests.py — do not edit. Expected bytes from AsmJit's",
       f"//! asmjit_test_assembler_x64.cpp: {total} cases, {len(cases)} ported, {total-len(cases)} unsupported.",
       "const std = @import(\"std\");", "const opts = @import(\"zigjit_options\");",
       "const CodeBuffer = @import(\"core/code_buffer.zig\").CodeBuffer;",
       "const x = @import(\"x86_64/assembler.zig\");", "const gp = @import(\"x86_64/gp.zig\");", "const vec = @import(\"x86_64/vec.zig\");", "",
       "fn check3(expected: []const u8, store: bool, acc: gp.Reg, addr: u64, text: []const u8) u32 {",
       "    var cb = CodeBuffer.init(std.testing.allocator);", "    defer cb.deinit();",
       "    gp.movMoffs(&cb, store, acc, addr) catch |e| {", "        std.debug.print(\"{s}: {s}\\n\", .{ @errorName(e), text });", "        return 1;", "    };",
       "    if (std.mem.eql(u8, expected, cb.code())) return 0;",
       "    std.debug.print(\"encoding mismatch: {s}: got {X}\\n\", .{ text, cb.code() });", "    return 1;", "}", "",
       "fn check(expected: []const u8, comptime mn: anytype, args: anytype, opt: gp.Opt, text: []const u8) u32 {",
       "    var cb = CodeBuffer.init(std.testing.allocator);", "    defer cb.deinit();",
       "    gp.emitOpt(&cb, mn, args, opt) catch |e| {", "        std.debug.print(\"{s}: {s}\\n\", .{ @errorName(e), text });", "        return 1;", "    };",
       "    if (std.mem.eql(u8, expected, cb.code())) return 0;",
       "    std.debug.print(\"encoding mismatch: {s}: expected {x}, got {x}\\n\", .{ text, expected, cb.code() });",
       "    return 1;", "}", "",
       "fn checkV(expected: []const u8, comptime mn: anytype, args: anytype, e: @import(\"x86_64/vec.zig\").Extra, text: []const u8) u32 {",
       "    var cb = CodeBuffer.init(std.testing.allocator);", "    defer cb.deinit();",
       "    vec.emit(&cb, mn, args, e) catch |err| {", "        std.debug.print(\"{s}: {s}\\n\", .{ @errorName(err), text });", "        return 1;", "    };",
       "    if (std.mem.eql(u8, expected, cb.code())) return 0;",
       "    std.debug.print(\"encoding mismatch: {s}: got {X}\\n\", .{ text, cb.code() });",
       "    return 1;", "}", ""]
N = 400
for i in range(0, len(cases), N):
    out += [f'test "AsmJit x86_64 assembler tests ({i}..{min(i+N, len(cases))})" {{',
            "    if (!opts.x86_64) return error.SkipZigTest;", "    @setEvalBranchQuota(1_000_000);",

            "    var fails: u32 = 0;"] + cases[i:i+N] + ["    try std.testing.expectEqual(@as(u32, 0), fails);", "}", ""]
(root / "src/asmjit_x64_tests.zig").write_text("\n".join(out))
rep = [f"total {total}, ported {len(cases)}, unsupported {total-len(cases)}"] + [f"{v:6} {k}" for k, v in unsup.most_common() if not k.startswith("  ")]
rep += ["", "top unsupported details:"] + [f"{v:6} {k.strip()}" for k, v in unsup.most_common() if k.startswith("  ")][:60]
rep += ["", "cases:"] + unsup_cases
(root / "bench/asmjit_x64_unsupported.txt").write_text("\n".join(rep) + "\n")
print("\n".join(rep[:30]))
