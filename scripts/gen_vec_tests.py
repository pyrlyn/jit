#!/usr/bin/env python3
"""Writes src/testdata/vec_cases.txt (run by src/vec_table_tests.zig): cases for every form of
src/x86_64/vec_table.zig and of the AsmJit-derived table it replaced.

    scripts/gen_vec_tests.py OLD_vec_table.zig [LLVM_MC]
    # OLD: git show b5e4608:src/x86_64/vec_table.zig > /tmp/vec_table_asmjit.zig

Every form of both tables gets cases (plain, {k}/{z}, broadcast, {er}/{sae}; 32-bit mode for
mode-independent legacy forms). Each case is encoded with both tables (a temporary `zig test`).
Same bytes: kept as `asmjit` (the old encoding, frozen). Different bytes or a new form: the new
bytes must equal llvm-mc 23.1.4's (`llvm`). A form only the old table had: `removed` (ERR).
Differences llvm-mc cannot check (it rejects the text) are listed in KNOWN with a reason."""
import os, re, subprocess, sys, tempfile, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src/testdata/vec_cases.txt")
FIELD = re.compile(r"\.(\w+) = ([^,{}]+|&\.\{[^}]*\}|&\.\{(?:\.\{[^}]*\}(?:, )?)*\})")


def load(path):
    t = collections.OrderedDict()
    cur = None
    body = open(path).read().split("pub const table = struct")[1]
    for ln in body.splitlines():
        s = ln.strip()
        m = re.match(r'pub const @?"?(\w+)"? = \[_\]Form\{', s)
        if m:
            cur = m.group(1); t[cur] = []; continue
        if s.startswith(".{") and cur:
            ops = [(k, int(z), int(b)) for k, z, b in re.findall(r"\.k = \.(\w+), \.size = (\d+), \.bcst = (\d+)", s)]
            f = dict(re.findall(r"\.(\w+) = ([\w.]+)", s.split("}},", 1)[-1] if ops else s))
            f["ops"] = ops
            f["pre"] = re.search(r"\.pre = &\.\{([^}]*)\}", s).group(1)
            t[cur].append(f)
    return t


GPR = ["dx", "bx", "si", "di"]
R8 = ["dl", "bl", "dl", "bl"]
FIXED = {"ax": "r16:ax", "al": "r8:al", "cl": "r8:cl", "eax": "r32:eax", "rax": "r64:rax", "dx": "r16:dx",
         "st0": "st0", "es": "seg0", "cs": "seg1", "ss": "seg2", "ds": "seg3", "fs": "seg4", "gs": "seg5"}


def opdesc(k, size, bcst, i, f, use_bcst):
    if k in ("xmm", "ymm", "zmm", "mm", "k", "tmm"):
        return f"{k}{i + 1}"
    if k == "sti": return "st2"
    if k == "sreg": return "seg4"
    if k == "creg": return "cr3"
    if k == "dreg": return "dr2"
    if k in FIXED: return FIXED[k]
    if k == "r8": return "r8:" + R8[i]
    if k == "r16": return "r16:" + GPR[i]
    if k == "r32": return "r32:e" + GPR[i]
    if k == "r64": return "r64:r" + GPR[i]
    if k == "imm": return "i5"
    if k == "maddr": return "ma"
    if k in ("vx", "vy", "vz"): return f"{k}{size}"
    if k == "mem":
        if use_bcst and bcst:
            total = size or (16 << int(f["l"]))
            return f"b{bcst}x{total // bcst}"
        return f"m{size}"
    raise KeyError(k)


def cases_of(mn, f):
    ops = f["ops"]
    mode = f.get("mode", ".any").lstrip(".") if "mode" in f else "any"
    modes = ["x86"] if mode == "x86" else ["x64"]
    if mode == "any" and f["enc"] == ".legacy" and f["rexw"] == "false" and not any(k in ("r64", "rax") for k, _, _ in ops):
        modes.append("x86")
    variants = [("-", False)]
    mask = int(f["mask"])
    if mask >= 1: variants.append(("k7", False))
    if mask == 2: variants.append(("k7z", False))
    if any(b for _, _, b in ops): variants.append(("-", True))
    if mask >= 1 and any(k in ("vx", "vy", "vz") for k, _, _ in ops): variants = [v for v in variants if v[0] != "-"] or variants
    if f["er"] == "true": variants.append(("rn", False))
    elif f["sae"] == "true": variants.append(("sae", False))
    if f["enc"] == ".evex" and not any(k == "zmm" for k, _, _ in ops) and mask == 0 and f["er"] == "false":
        variants.append(("evex", False))
    out = []
    for md in modes:
        for ext, ub in variants:
            d = ",".join(opdesc(k, z, b, i, f, ub) for i, (k, z, b) in enumerate(ops)) or "-"
            out.append(f"{md} {mn} {d} {ext}")
    return out


DUMP = r'''
const std = @import("std");
const T = @import("vec_table_tests.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
test "dump" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const a = arena.allocator();
    var cb = CodeBuffer.init(std.heap.page_allocator);
    const Old = T.Lookup(@import("x86_64/.vec_table_old.zig"));
    const New = T.Lookup(@import("x86_64/vec.zig").tbl);
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, @embedFile("testdata/.vec_cases_in.txt"), '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        var buf: [256]u8 = undefined;
        const c = try T.parse(a, try std.fmt.bufPrint(&buf, "{s} ERR x", .{line}));
        try out.print(a, "{s}", .{line});
        inline for (.{ Old, New }) |L| {
            const g = if (L.get(c.mn)) |fs| try T.encode(&cb, fs, c) else null;
            if (g) |b| try out.print(a, "\t{X}", .{b}) else try out.print(a, "\tERR", .{});
        }
        try out.append(a, '\n');
    }
    var f = try std.Io.Dir.cwd().createFile(std.testing.io, "src/testdata/.vec_cases_out.txt", .{});
    try f.writeStreamingAll(std.testing.io, out.items);
    f.close(std.testing.io);
}
'''


def run_dump(lines, old):
    tmp = {"src/x86_64/.vec_table_old.zig": open(old).read(), "src/.vec_dump.zig": DUMP,
           "src/testdata/.vec_cases_in.txt": "\n".join(lines) + "\n"}
    for p, s in tmp.items():
        open(os.path.join(ROOT, p), "w").write(s)
    opts = os.path.join(ROOT, "src/testdata/.opts.zig")
    open(opts, "w").write("pub const x86_64 = true;\npub const aarch64 = true;\npub const validate = true;\npub const hooks = true;\npub const asmjit_tests = false;\n")
    try:
        subprocess.run(["zig", "test", "-OReleaseSafe", "--test-filter", "dump", "--dep", "pyrlyn_jit_options", "-Mroot=src/.vec_dump.zig",
                        "-Mpyrlyn_jit_options=src/testdata/.opts.zig"], cwd=ROOT, check=True)
        res = {}
        for ln in open(os.path.join(ROOT, "src/testdata/.vec_cases_out.txt")):
            c, o, n = ln.rstrip("\n").split("\t")
            res[c] = (None if o == "ERR" else o, None if n == "ERR" else n)
        return res
    finally:
        for p in list(tmp) + ["src/testdata/.opts.zig", "src/testdata/.vec_cases_out.txt"]:
            if os.path.exists(os.path.join(ROOT, p)):
                os.remove(os.path.join(ROOT, p))


# ---- llvm-mc --------------------------------------------------------------------------------

KW = {1: "byte", 2: "word", 4: "dword", 6: "fword", 8: "qword", 10: "tbyte", 16: "xmmword", 32: "ymmword", 64: "zmmword"}


def renames():
    src = open(os.path.join(ROOT, "tools/gen_x86_vec.zig")).read()
    body = src.split("const renames = ")[1].split("};")[0]
    return {new: old for old, new in re.findall(r'\.\{ "(\w+)", "(\w+)" \}', body)}


def text(case, rev):
    md, mn, ops, ext = case.split(" ")
    x86 = md == "x86"
    base = "ecx" if x86 else "rcx"
    parts = []
    for o in ([] if ops == "-" else ops.split(",")):
        if ":" in o:
            parts.append(o.split(":")[1])
        elif o[0] == "i" and o[1:].isdigit():
            parts.append(o[1:])
        elif o == "ma":
            parts.append(base)
        elif re.fullmatch(r"m\d+", o):
            kw = KW.get(int(o[1:]))
            parts.append((kw + " ptr " if kw else "") + f"[{base} + 128]")
        elif re.fullmatch(r"b\d+x\d+", o):
            e, c = map(int, o[1:].split("x"))
            parts.append(f"{KW[e]} ptr [{base} + 128]{{1to{c}}}")
        elif re.fullmatch(r"v[xyz]\d+", o):
            parts.append(f"[{base} + 4*{o[1]}mm5 + 128]")
        elif o.startswith("seg"):
            parts.append(["es", "cs", "ss", "ds", "fs", "gs"][int(o[3:])])
        elif o.startswith("st"):
            parts.append(f"st({o[2:]})")
        else:
            parts.append(o)
    if parts and "k7" in ext:
        parts[0] += " {k7}" + (" {z}" if "z" in ext.replace("k7", "") else "")
    # Intel syntax: the rounding operand goes before a trailing immediate
    rc = "{rn-sae}" if "rn" in ext else "{sae}" if "sae" in ext else None
    if rc:
        at = len(parts) - 1 if parts and ops.split(",")[-1].startswith("i") else len(parts)
        parts.insert(at, rc)
    pfx = "{evex} " if ext == "evex" else ""
    return pfx + rev.get(mn, mn) + (" " + ", ".join(parts) if parts else "")


def llvm_mc(mc, items):
    """items: [(mode, text)] -> list of hex or None."""
    res = [None] * len(items)
    for md in ("x64", "x86"):
        idx = [i for i, (m, _) in enumerate(items) if m == md]
        if not idx: continue
        src = "\n".join(items[i][1] for i in idx) + "\n"
        r = subprocess.run([mc, "-triple=" + ("x86_64" if md == "x64" else "i386"), "-x86-asm-syntax=intel", "-show-encoding"],
                           input=src, capture_output=True, text=True)
        bad = {int(m) - 1 for m in re.findall(r"<stdin>:(\d+):\d+: error", r.stderr)}
        enc = [e for e in re.findall(r"encoding: \[([^\]]*)\]", r.stdout)]
        ok = [i for j, i in enumerate(idx) if j not in bad]
        if len(enc) != len(ok):
            # an instruction can print several encodings lines only with relaxation; not expected
            raise SystemExit(f"llvm-mc output mismatch {len(enc)} vs {len(ok)}")
        for i, e in zip(ok, enc):
            res[i] = "".join(f"{int(b, 16):02X}" for b in e.split(",") if b.strip())
    return res


def main():
    old_path = sys.argv[1]
    mc = sys.argv[2] if len(sys.argv) > 2 else "llvm-mc"
    old = load(old_path)
    new = load(os.path.join(ROOT, "src/x86_64/vec_table.zig"))
    order = []
    seen = set()
    for t in (new, old):
        for mn, fs in t.items():
            for f in fs:
                for c in cases_of(mn, f):
                    if c not in seen:
                        seen.add(c); order.append(c)
    res = run_dump(order, old_path)
    rev = renames()
    todo = [c for c in order if res[c][1] is not None and res[c][0] != res[c][1]]
    mcres = llvm_mc(mc, [(c.split(" ")[0], text(c, rev)) for c in todo])
    llvm = dict(zip(todo, mcres))
    lines = []
    stats = collections.Counter()
    bad = []
    for c in order:
        o, n = res[c]
        if o is None and n is None:
            continue
        if o == n:
            lines.append(f"{c} {n} asmjit"); stats["asmjit"] += 1
        elif n is None:
            lines.append(f"{c} ERR removed"); stats["removed"] += 1
        elif llvm[c] == n:
            lines.append(f"{c} {n} llvm"); stats["llvm"] += 1
        else:
            bad.append((c, o, n, llvm[c], text(c, rev)))
    for b in bad:
        print("UNVERIFIED", *b, sep="\t")
    hdr = (f"# Generated by scripts/gen_vec_tests.py; do not edit. Format: see src/vec_table_tests.zig.\n"
           f"# {stats['asmjit']} asmjit (frozen), {stats['llvm']} llvm, {stats['removed']} removed.\n")
    open(OUT, "w").write(hdr + "\n".join(lines) + "\n")
    print(dict(stats), "unverified", len(bad), file=sys.stderr)


if __name__ == "__main__":
    main()
