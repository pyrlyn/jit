# bc2jit C API benchmark

Host: macOS-27.0.1-arm64-arm-64bit-Mach-O; Apple M3 Max. Zig 0.17.0.
ReleaseFast, default backends and hooks, validation off, unstripped artifacts. Identical source/options except `-Dbc2jit`.
7 runs per build, alternating order; median of per-run medians. Each run: 200 warmups, 15 samples × 500 iterations (cached calls: 100000).

## Cost of enabling support

Same assembler C API workload, no KXBC calls in the timed regions.

| Metric (ns/op) | Disabled | Enabled | Delta |
|---|---:|---:|---:|
| direct_compile_ns | 2530.000 | 2512.000 | -0.71% |
| direct_jit_ns | 2684.000 | 2668.000 | -0.60% |
| direct_call_ns | 0.830 | 0.840 | +1.20% |

## Assembler versus bytecode API

Both paths compile `add(a,b)` and check result 42. KXBC v0 emits byte-identical code to the assembler path (checked by the benchmark); v1 uses direct lowering for compatible straight-line code and the IR compiler otherwise.
The assembler C API performs mnemonic/form lookup for each instruction; KXBC v0 uses specialized Zig emitters. These numbers do not compare KXBC with direct Zig assembler calls.
`compile`: allocate, emit/decode+lower, finalize, free; no executable mapping. `jit`: same plus acquire an executable mapping, one call, release ownership to the bounded runtime cache. `call`: cached function only (includes loop, indirect call and volatile result store).

| Metric (ns/op) | Assembler | KXBC v0 | KXBC v1 |
|---|---:|---:|---:|
| compile | 2512.000 | 120.000 | 212.000 |
| jit | 2668.000 | 258.000 | 352.000 |
| cold_jit | 9630.000 | 7216.000 | 7292.000 |
| call | 0.840 | 0.830 | 0.840 |

The runtime cache is shared by both paths and retains at most 128 KiB. Compilation still decodes and emits fresh code on every iteration; no function/result cache is used. `cold_jit` also trims all idle OS mappings every iteration. The 2× requirement applies to the warmed repeated JIT lifecycle, not forced unmapping or cached execution.

Generated code: v0 20 bytes, v1 20 bytes.

## Artifact size

Sizes on disk in bytes; executable statically links the same C API benchmark in both builds. Disabled builds retain only bytecode API stubs; both binaries reference those symbols.

| Artifact | Disabled | Enabled | Delta bytes | Delta |
|---|---:|---:|---:|---:|
| bin/bench_bc2jit_api | 1,223,392 | 1,609,056 | +385,664 | +31.52% |
| bin/bench_x86_64 | 534,816 | 534,816 | +0 | +0.00% |
| bin/example_aarch64 | 401,024 | 401,024 | +0 | +0.00% |
| bin/example_hooks | 416,832 | 416,832 | +0 | +0.00% |
| bin/example_x86_64 | 417,328 | 417,328 | +0 | +0.00% |
| lib/libpyrlyn-jit.a | 5,506,928 | 5,768,520 | +261,592 | +4.75% |
| lib/libpyrlyn-jit.dylib | 1,862,624 | 1,882,368 | +19,744 | +1.06% |

## Per-run ranges

Min–max of the per-run median (ns/op); small timing differences can be measurement/layout noise, especially mapping syscalls.

| Metric | Disabled min–max | Enabled min–max |
|---|---:|---:|
| direct_compile_ns | 2508.000–2576.000 | 2464.000–2560.000 |
| direct_jit_ns | 2660.000–2720.000 | 2608.000–2796.000 |
| direct_cold_jit_ns | 9326.000–10934.000 | 9158.000–10858.000 |
| direct_call_ns | 0.810–0.840 | 0.810–0.840 |
| bc_v0_compile_ns | — | 116.000–126.000 |
| bc_v0_jit_ns | — | 254.000–286.000 |
| bc_v0_cold_jit_ns | — | 6826.000–8124.000 |
| bc_v0_call_ns | — | 0.830–0.860 |
| bc_v1_compile_ns | — | 206.000–216.000 |
| bc_v1_jit_ns | — | 344.000–370.000 |
| bc_v1_cold_jit_ns | — | 6554.000–8128.000 |
| bc_v1_call_ns | — | 0.830–0.940 |

This is a small-function microbenchmark, not an application throughput estimate. This arithmetic fixture uses direct lowering in both bytecode versions; v1 control flow, locals, deep stacks and extended operations still use the IR compiler.

Reproduce: `python3 scripts/bench_bc2jit_api.py --zig /path/to/zig --report bench/BC2JIT_API_RESULTS.md`.
Raw samples: `zig-out/bc2jit-compare/results.json`. This script does not rewrite regression baselines.

## Full-cycle acceptance

Threshold: 2.00×, against the assembler C API in the disabled build, with the same runtime cache in both builds.
- KXBC v0: median 10.40×; minimum paired run 9.37×
- KXBC v1: median 7.62×; minimum paired run 7.28×
Result: PASS.

## Validation

Correctness and formatting passed after integrating C API 0.2.0 (main `f4fc391`): native full tests with bytecode enabled/disabled, x86-64 API/runtime tests under Rosetta, LLVM-enabled full tests, and installed static/shared C consumers in both feature settings.

The three runtime-related baseline values were tightened using the median of three run medians: `jit_runtime_add_release` 173.3 ns, `e2e_aarch64_compile_run` 197.4 ns, `bc2jit_e2e_aarch64` 1048.5 ns. These gates passed on a subsequent run. Other baselines and tolerances were not changed.

The existing overall `bench-test` gate remains non-green for: `aarch64_emit`, `bc2jit_decode_64k`. The persistent `aarch64_emit` and `bc2jit_decode_64k` baseline failures were also reproduced on the original `20d0ed4` control before these API/runtime changes; transient unrelated thresholds vary with host load.
