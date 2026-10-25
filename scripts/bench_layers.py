#!/usr/bin/env python3
"""Compare safety-layer configurations against AsmJit, runs interleaved to reduce drift.

Expects binaries in bench/bins/ (baseline = pre-safety build, off, validate, validate_hooks),
built with `zig build -Doptimize=ReleaseFast [-Dvalidate=..] [-Dhooks=..]`, and bench/asmjit_bench.
usage: python3 scripts/bench_layers.py [runs] [iters]
"""
import os, re, statistics, subprocess, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 9
iters = sys.argv[2] if len(sys.argv) > 2 else "5000"
cfgs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["baseline", "off", "validate", "validate_hooks"]
data = {}
for _ in range(runs):
    jobs = [("asmjit", [os.path.join(root, "bench/asmjit_bench"), iters, "/tmp"], {"ASMJIT_OPT_SIZE": "1"})]
    # "name+cb" runs bench/bins/name with ZJ_HOOK=1 (attach the runtime counting callback).
    jobs += [(c, [os.path.join(root, "bench/bins", c.split("+")[0]), iters], {"ZJ_HOOK": "1"} if c.endswith("+cb") else {}) for c in cfgs]
    for name, cmd, env in jobs:
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
        for ln in (p.stdout + p.stderr).splitlines():
            m = re.match(r"(\w+)\s+(\w+)\s.*avg_ns_per_instr=([\d.]+)", ln)
            if m:
                data.setdefault((name, m.group(2)), []).append(float(m.group(3)))
cols = ["asmjit"] + cfgs
print("| scenario | " + " | ".join(cols) + " |")
print("|---|" + "--:|" * len(cols))
for s in ("straight", "calls", "loops", "mem"):
    print(f"| {s} | " + " | ".join(f"{statistics.median(data[(c, s)]):.2f}" for c in cols) + " |")
