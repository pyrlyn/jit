#!/usr/bin/env python3
"""Compare identical ReleaseFast builds with bc2jit on/off, time C API paths and size artifacts."""
import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--report", type=Path, help="Write the Markdown report here")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    root = Path(__file__).resolve().parents[1]
    output = root / "zig-out" / "bc2jit-compare"
    samples = {"on": [], "off": []}
    sizes = {}
    for mode in samples:
        prefix = output / mode
        subprocess.run([args.zig, "build", "install", "bench-bc2jit-api", "-Doptimize=ReleaseFast",
                        f"-Dbc2jit={'true' if mode == 'on' else 'false'}", "--prefix", str(prefix), "-j4"],
                       cwd=root, check=True)
        sizes[mode] = {str(p.relative_to(prefix)): p.stat().st_size
                       for folder in ("bin", "lib") for p in (prefix / folder).glob("*") if p.is_file()}
    # No concurrent builds or benchmarks. Alternate order to reduce drift bias.
    for i in range(args.runs):
        for mode in (("off", "on") if i % 2 == 0 else ("on", "off")):
            exe = output / mode / "bin" / ("bench_bc2jit_api.exe" if platform.system() == "Windows" else "bench_bc2jit_api")
            values = dict(line.split() for line in run(str(exe)).splitlines())
            samples[mode].append({k: float(v) for k, v in values.items()})
    result = {"platform": platform.platform(), "zig": run(args.zig, "version"), "runs": args.runs,
              "samples": samples, "sizes_bytes": sizes}
    if platform.system() == "Darwin":
        result["cpu"] = run("sysctl", "-n", "machdep.cpu.brand_string")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def median(mode, key):
        return statistics.median(s[key] for s in samples[mode])

    lines = ["# bc2jit C API benchmark", "", f"Host: {result['platform']}; {result.get('cpu', platform.processor())}. Zig {result['zig']}.",
             "ReleaseFast, default backends and hooks, validation off, unstripped artifacts. Identical source/options except `-Dbc2jit`.",
             f"{args.runs} runs per build, alternating order; median of per-run medians. Each run: 200 warmups, 15 samples × 500 iterations (cached calls: 100000).",
             "", "## Cost of enabling support", "", "Same assembler C API workload, no KXBC calls in the timed regions.", "",
             "| Metric (ns/op) | Disabled | Enabled | Delta |", "|---|---:|---:|---:|"]
    for key in ("direct_compile_ns", "direct_jit_ns", "direct_call_ns"):
        off, on = median("off", key), median("on", key)
        lines.append(f"| {key} | {off:.3f} | {on:.3f} | {(on/off-1)*100:+.2f}% |")
    lines += ["", "## Assembler versus bytecode API", "", "Both paths compile `add(a,b)` and check result 42. KXBC v0 emits byte-identical code to the assembler path (checked by the benchmark); v1 uses the IR compiler.",
              "The assembler C API performs mnemonic/form lookup for each instruction; KXBC v0 uses specialized Zig emitters. These numbers do not compare KXBC with direct Zig assembler calls.",
              "`compile`: allocate, emit/decode+lower, finalize, free; no executable mapping. `jit`: same plus map, one call, release. `call`: cached function only (includes loop, indirect call and volatile result store).",
              "", "| Metric (ns/op) | Assembler | KXBC v0 | KXBC v1 |", "|---|---:|---:|---:|"]
    for metric in ("compile", "jit", "call"):
        lines.append(f"| {metric} | {median('on', f'direct_{metric}_ns'):.3f} | {median('on', f'bc_v0_{metric}_ns'):.3f} | {median('on', f'bc_v1_{metric}_ns'):.3f} |")
    lines += ["", f"Generated code: v0 {median('on', 'bc_v0_code_bytes'):.0f} bytes, v1 {median('on', 'bc_v1_code_bytes'):.0f} bytes.",
              "", "## Artifact size", "", "Sizes on disk in bytes; executable statically links the same C API benchmark in both builds. Disabled builds retain only bytecode API stubs; both binaries reference those symbols.",
              "", "| Artifact | Disabled | Enabled | Delta bytes | Delta |", "|---|---:|---:|---:|---:|"]
    for name in sorted(sizes["off"].keys() & sizes["on"].keys()):
        off, on = sizes["off"][name], sizes["on"][name]
        lines.append(f"| {name} | {off:,} | {on:,} | {on-off:+,} | {(on/off-1)*100:+.2f}% |")
    lines += ["", "## Per-run ranges", "", "Min–max of the per-run median (ns/op); small timing differences can be measurement/layout noise, especially mapping syscalls.",
              "", "| Metric | Disabled min–max | Enabled min–max |", "|---|---:|---:|"]
    for key in samples['on'][0]:
        if not key.endswith('_ns'):
            continue
        def span(mode):
            values = [s[key] for s in samples[mode] if key in s]
            return f"{min(values):.3f}–{max(values):.3f}" if values else "—"
        lines.append(f"| {key} | {span('off')} | {span('on')} |")
    lines += ["", "This is a small-function microbenchmark, not an application throughput estimate. The v1 result includes fixed IR construction/register-allocation costs.",
              "", "Reproduce: `python3 scripts/bench_bc2jit_api.py --zig /path/to/zig --report bench/BC2JIT_API_RESULTS.md`.",
              "Raw samples: `zig-out/bc2jit-compare/results.json`. Existing regression baselines are unchanged.", ""]
    report = "\n".join(lines)
    if args.report:
        args.report.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
