#!/usr/bin/env python3
"""Compare two compile_bench executables, alternating order to reduce timing drift.

Both binaries must use the current benchmark harness (checksum support), built
against the respective library revisions. No third-party Python dependencies.
"""
import argparse
import math
import os
import re
import statistics
import subprocess

ROW = re.compile(r"^(aarch64|x86_64|riscv64)\s+(small|medium|large|huge)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)$")


def run(binary, ms, seed):
    env = {**os.environ, "PJIT_BENCH_MS": str(ms), "PJIT_BENCH_SEED": str(seed), "PJIT_BENCH_CHECKSUM": "1"}
    env.pop("PJIT_BENCH_ONLY", None)
    p = subprocess.run([binary], env=env, text=True, capture_output=True, check=True)
    rows, hashes = {}, {}
    for line in (p.stdout + p.stderr).splitlines():
        if m := ROW.match(line):
            rows[f"{m[1]}/{m[2]}"] = tuple(map(float, m.groups()[2:]))
        elif line.startswith("checksum "):
            _, key, digest = line.split()
            hashes[key] = digest
    if len(rows) != 12 or rows.keys() != hashes.keys():
        raise RuntimeError(f"{binary}: expected all 12 cases and checksums; got:\n{p.stdout}{p.stderr}")
    return rows, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ms", type=int, default=300)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0x5EED)
    args = parser.parse_args()
    if args.repeats < 1 or args.ms < 0:
        parser.error("repeats must be positive and ms nonnegative")
    measurements = [[], []]
    hashes = None
    for repeat in range(args.repeats):
        for side in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            rows, current = run((args.baseline, args.candidate)[side], args.ms, args.seed)
            if hashes is not None and current != hashes:
                raise RuntimeError("machine-code checksums differ between builds/runs")
            hashes = current
            measurements[side].append(rows)
    print(f"{args.repeats} alternating runs, {args.ms} ms/case/run, seed {args.seed}; all code checksums match.\n")
    print("| Backend/case | Before µs | After µs | Time reduction | Before peak KiB | After peak KiB | Peak reduction |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    ratios = [[], []]
    for key in measurements[0][0]:
        before, after = [tuple(statistics.median(run[key][i] for run in side) for i in range(2)) for side in measurements]
        time_ratio, memory_ratio = (after[i] / before[i] for i in range(2))
        ratios[0].append(time_ratio)
        ratios[1].append(memory_ratio)
        print(f"| {key} | {before[0]:.3f} | {after[0]:.3f} | {(1-time_ratio)*100:.1f}% | {before[1]:.1f} | {after[1]:.1f} | {(1-memory_ratio)*100:.1f}% |")
    reduction = [(1 - math.exp(statistics.mean(map(math.log, values)))) * 100 for values in ratios]
    print(f"\nEqual-weight geometric mean across all 12 cases: time −{reduction[0]:.1f}%, peak allocated heap −{reduction[1]:.1f}%.")
    print("These are suite averages, not a minimum improvement for each individual case. Peak heap excludes input IR, stack and allocator overhead/RSS.")


if __name__ == "__main__":
    main()
