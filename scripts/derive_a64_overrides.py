#!/usr/bin/env python3
"""Derive fixed-bit corrections for the AArch64 JSON DB from AsmJit's expected encodings.

Reads a test log (`zig test src/asmjit_a64_tests.zig`) and, for every "encoding mismatch ...
form=<mn>#<i>" whose difference lies entirely in bits the form keeps fixed (no operand field),
records an XOR mask for that form. Conflicting masks for the same form are dropped.
Writes scripts/a64_overrides.json (merged with the existing one).

    python3 scripts/derive_a64_overrides.py /tmp/a64.txt
"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_a64_tables as g

table = g.main(write=False)
log = open(sys.argv[1]).read()
masks = collections.defaultdict(set)
hits = [(int.from_bytes(bytes.fromhex(a), "little"), int.from_bytes(bytes.fromhex(b), "little"), mn, idx)
        for a, b, mn, idx in re.findall(r"encoding mismatch: ([0-9A-F]{8}) .*?: got ([0-9A-F]{8}) form=([\w.]+)#(\d+)", log)]
# SVE log (scripts/gen_sve_tests.py): words are printed as numbers.
hits += [(int(a, 16), int(b, 16), mn, idx) for a, b, mn, idx in re.findall(r"want ([0-9A-F]{8}) got ([0-9A-F]{8}) form=([\w.]+)#(\d+)", log)]
for e, o, mn, idx in hits:
    if int(idx) >= 1000: continue  # assembler-level alias (a64.encodeMnIdx), not a table form
    f = table[mn][int(idx)]
    fmask = 0
    for _, at, w, _ in f["fields"]: fmask |= ((1 << w) - 1) << at
    d = e ^ o
    if d & 0x40000000 and f.get("t"): d = None  # Q with an arrangement list: fix the list instead
    masks[f["key"]].add(d if d is not None and d & fmask == 0 else None)
ovr = json.load(open(g.OVR)) if os.path.exists(g.OVR) else {}
added = 0
for k, ms in masks.items():
    if len(ms) == 1 and None not in ms:
        m = ms.pop()
        ovr[k] = f"{int(ovr.get(k, '0'), 16) ^ m:08X}"; added += 1
json.dump(dict(sorted(ovr.items())), open(g.OVR, "w"), indent=1)
print(f"{added} forms patched, {sum(1 for v in masks.values() if len(v) != 1 or None in v)} not patchable")
