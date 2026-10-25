#!/usr/bin/env python3
"""Writes src/simd_tests.zig: x86-64 SSE/AVX (src/x86_64/simd.zig) and AArch64 NEON
(src/aarch64/neon.zig) encodings checked byte-for-byte against LLVM (clang -c + objdump)."""
import os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_gp_tests import assemble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMD = open(os.path.join(ROOT, "src/x86_64/simd.zig")).read()
OPS = re.findall(r"pub const (\w+): Op = \.\{([^}]*)\}", SIMD)
MEMS = [("x.ptr(.rax, 0)", "[rax]"), ("x.ptrIdx(.r12, .rcx, .x4, 16)", "[r12+4*rcx+16]"),
        ("x.ptr(.rbp, -200)", "[rbp-200]"), ("x.ptrIdx(.rsp, .r9, .x8, 0x1000)", "[rsp+8*r9+4096]")]
REGS = [(1, 2, 3), (9, 3, 14), (2, 12, 7), (15, 8, 0)]


def x86_cases():
    out = []
    for name, body in OPS:
        st, two, vex_only = name.endswith("_st"), ".two = true" in body, "vex = .fma" in body or "vex = .avx2" in body
        mn = name[:-3] if st else name
        bx = "src_xmm" in body
        sc = re.search(r"\.scalar = (\d)", body)
        sp = {"4": "dword", "8": "qword"}[sc.group(1)] if sc else "xmmword"
        if not vex_only:
            for d, s, _ in REGS:
                if not st:
                    out.append((f"try s.sse(&cb, s.ops.{name}, s.xmm({d}), s.r(s.xmm({s})));", f"{mn} xmm{d}, xmm{s}"))
            for (z, t), (d, _, _) in zip(MEMS, REGS):
                txt = f"{mn} xmmword ptr {t}, xmm{d}" if st else f"{mn} xmm{d}, {sp} ptr {t}"
                out.append((f"try s.sse(&cb, s.ops.{name}, s.xmm({d}), s.m({z}));", txt))
        vmn = mn if mn.startswith("v") else "v" + mn
        for wide in (False, True):
            if wide and sc:
                continue
            V, P = ("ymm", "ymmword") if wide else ("xmm", "xmmword")
            S = "xmm" if bx else V
            SP = "xmmword" if bx else P
            for d, a, b in REGS:
                if st:
                    continue
                if two:
                    out.append((f"try s.vex(&cb, s.ops.{name}, s.{V}({d}), null, s.r(s.{S}({b})));", f"{vmn} {V}{d}, {S}{b}"))
                else:
                    out.append((f"try s.vex(&cb, s.ops.{name}, s.{V}({d}), s.{V}({a}), s.r(s.{V}({b})));", f"{vmn} {V}{d}, {V}{a}, {V}{b}"))
            for (z, t), (d, a, _) in zip(MEMS, REGS):
                mp = "dword" if bx else P
                if two:
                    txt = f"{vmn} {mp} ptr {t}, {V}{d}" if st else f"{vmn} {V}{d}, {mp} ptr {t}"
                    out.append((f"try s.vex(&cb, s.ops.{name}, s.{V}({d}), null, s.m({z}));", txt))
                else:
                    out.append((f"try s.vex(&cb, s.ops.{name}, s.{V}({d}), s.{V}({a}), s.m({z}));", f"{vmn} {V}{d}, {V}{a}, {sp if sc else P} ptr {t}"))
    out.append(("try s.vzeroupper(&cb);", "vzeroupper"))
    return out


NEON = [  # (zig, llvm)
    ("try n.add(&cb, .s4, 0, 1, 2);", "add v0.4s, v1.4s, v2.4s"), ("try n.add(&cb, .b16, 31, 7, 9);", "add v31.16b, v7.16b, v9.16b"),
    ("try n.add(&cb, .d2, 3, 4, 5);", "add v3.2d, v4.2d, v5.2d"), ("try n.add(&cb, .h8, 3, 4, 5);", "add v3.8h, v4.8h, v5.8h"),
    ("try n.sub(&cb, .s4, 8, 9, 10);", "sub v8.4s, v9.4s, v10.4s"), ("try n.mul(&cb, .s4, 1, 2, 3);", "mul v1.4s, v2.4s, v3.4s"),
    ("try n.mul(&cb, .h8, 1, 2, 3);", "mul v1.8h, v2.8h, v3.8h"),
    ("try n.fadd(&cb, .s4, 0, 1, 2);", "fadd v0.4s, v1.4s, v2.4s"), ("try n.fadd(&cb, .d2, 20, 21, 22);", "fadd v20.2d, v21.2d, v22.2d"),
    ("try n.fsub(&cb, .s4, 0, 1, 2);", "fsub v0.4s, v1.4s, v2.4s"), ("try n.fmul(&cb, .s4, 5, 6, 7);", "fmul v5.4s, v6.4s, v7.4s"),
    ("try n.fmul(&cb, .d2, 5, 6, 7);", "fmul v5.2d, v6.2d, v7.2d"), ("try n.fmla(&cb, .s4, 0, 1, 2);", "fmla v0.4s, v1.4s, v2.4s"),
    ("try n.andv(&cb, 0, 1, 2);", "and v0.16b, v1.16b, v2.16b"), ("try n.orr(&cb, 3, 4, 5);", "orr v3.16b, v4.16b, v5.16b"),
    ("try n.eor(&cb, 30, 29, 28);", "eor v30.16b, v29.16b, v28.16b"),
    ("try n.ldrQ(&cb, 0, .x1, 0);", "ldr q0, [x1]"), ("try n.ldrQ(&cb, 17, .zr_sp, 4080);", "ldr q17, [sp, #4080]"),
    ("try n.strQ(&cb, 2, .x3, 32);", "str q2, [x3, #32]"), ("try n.dupS(&cb, 4, .x5);", "dup v4.4s, w5"),
    ("try n.umovS(&cb, .x0, 7, 3);", "mov w0, v7.s[3]"), ("try n.movV(&cb, 1, 2);", "mov v1.16b, v2.16b"),
]


def neon_bytes(lines):
    with tempfile.TemporaryDirectory() as d:
        src, obj = os.path.join(d, "t.s"), os.path.join(d, "t.o")
        open(src, "w").write("\n".join(lines) + "\n")
        subprocess.run(["clang", "-c", "-target", "arm64-apple-macos", src, "-o", obj], check=True)
        out = subprocess.run(["objdump", "-d", obj], check=True, capture_output=True, text=True).stdout
    res = []
    for ln in out.splitlines():
        mm = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", ln)
        if mm:
            res.append(list(int(mm.group(1), 16).to_bytes(4, "little")))
    assert len(res) == len(lines), (len(res), len(lines))
    return res


def block(name, cond, imports, cases, got):
    o = [f'test "{name}" {{', f"    if (!{cond}) return error.SkipZigTest;", *imports,
         "    var cb = CodeBuffer.init(std.testing.allocator);", "    defer cb.deinit();", "    var fails: u32 = 0;"]
    for i, (z, t) in enumerate(cases):
        if i not in got:
            continue
        o += ["    cb.reset();", f"    {z}", f'    fails += check(&.{{ {", ".join(hex(b) for b in got[i])} }}, cb.code(), "{t}");']
    o += ["    try std.testing.expectEqual(@as(u32, 0), fails);", "}"]
    return o


def main():
    xc = x86_cases()
    xg = assemble([t for _, t in xc])
    missing = [xc[i][1] for i in range(len(xc)) if i not in xg]
    if missing:
        raise SystemExit("LLVM rejected: " + "; ".join(missing[:20]))
    ng = dict(enumerate(neon_bytes([t for _, t in NEON])))
    hdr = ["//! GENERATED by scripts/gen_simd_tests.py; do not edit. Expected bytes from LLVM.",
           f"//! {len(xg)} x86-64 SSE/AVX cases, {len(ng)} AArch64 NEON cases.",
           'const std = @import("std");', 'const opts = @import("zigjit_options");',
           'const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;', "",
           "fn check(expected: []const u8, got: []const u8, text: []const u8) u32 {",
           "    if (std.mem.eql(u8, expected, got)) return 0;",
           '    std.debug.print("simd mismatch: {s}: expected {x}, got {x}\\n", .{ text, expected, got });', "    return 1;", "}", ""]
    body = block("x86_64 SSE/AVX vs LLVM", "opts.x86_64", ['    const x = @import("x86_64/assembler.zig");', '    const s = @import("x86_64/simd.zig");'], xc, xg)
    body += [""] + block("aarch64 NEON vs LLVM", "opts.aarch64", ['    const n = @import("aarch64/neon.zig");'], NEON, ng)
    open(os.path.join(ROOT, "src/simd_tests.zig"), "w").write("\n".join(hdr + body) + "\n")
    print(f"{len(xg)} x86 + {len(ng)} neon cases")


if __name__ == "__main__":
    main()
