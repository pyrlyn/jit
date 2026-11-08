# bc2jit C API benchmark

Host: macOS-27.0.1-arm64-arm-64bit-Mach-O; Apple M3 Max. Zig 0.17.0.
ReleaseFast, default backends and hooks, validation off, unstripped artifacts. Identical source/options except `-Dbc2jit`.
7 runs per build, alternating order; median of per-run medians. Each run: 200 warmups, 15 samples × 500 iterations (cached calls: 100000).

## Cost of enabling support

Same assembler C API workload, no KXBC calls in the timed regions.

| Metric (ns/op) | Disabled | Enabled | Delta |
|---|---:|---:|---:|
| direct_compile_ns | 2508.000 | 2508.000 | +0.00% |
| direct_jit_ns | 2718.000 | 2746.000 | +1.03% |
| direct_call_ns | 0.830 | 0.830 | +0.00% |

## Assembler versus bytecode API

Both paths compile `add(a,b)` and check result 42. KXBC v0 emits byte-identical code to the assembler path (checked by the benchmark); v1 uses direct lowering for compatible straight-line code and the IR compiler otherwise.
The assembler C API performs mnemonic/form lookup for each instruction; KXBC v0 uses specialized Zig emitters. These numbers do not compare KXBC with direct Zig assembler calls.
`compile`: allocate, emit/decode+lower, finalize, free; no executable mapping. `jit`: same plus acquire an executable mapping, one call, release ownership to the bounded runtime cache. `call`: cached function only (includes loop, indirect call and volatile result store).

| Metric (ns/op) | Assembler | KXBC v0 | KXBC v1 |
|---|---:|---:|---:|
| compile | 2508.000 | 122.000 | 222.000 |
| jit | 2746.000 | 264.000 | 358.000 |
| cold_jit | 10764.000 | 8224.000 | 8110.000 |
| call | 0.830 | 0.830 | 0.810 |

The runtime cache is shared by both paths and retains at most 128 KiB. Compilation still decodes and emits fresh code on every iteration; no function/result cache is used. `cold_jit` also trims all idle OS mappings every iteration. The 2× requirement applies to the warmed repeated JIT lifecycle, not forced unmapping or cached execution.

Generated code: v0 20 bytes, v1 20 bytes.

## Artifact size

Sizes on disk in bytes; executable statically links the same C API benchmark in both builds. Disabled builds retain only bytecode API stubs; both binaries reference those symbols.

| Artifact | Disabled | Enabled | Delta bytes | Delta |
|---|---:|---:|---:|---:|
| bin/bench_bc2jit_api | 1,223,024 | 1,608,640 | +385,616 | +31.53% |
| bin/bench_x86_64 | 534,816 | 534,816 | +0 | +0.00% |
| bin/example_aarch64 | 401,024 | 401,024 | +0 | +0.00% |
| bin/example_hooks | 416,832 | 416,832 | +0 | +0.00% |
| bin/example_x86_64 | 417,328 | 417,328 | +0 | +0.00% |
| lib/libpyrlyn-jit.a | 1,742,344 | 4,883,000 | +3,140,656 | +180.25% |
| lib/libpyrlyn-jit.dylib | 1,349,216 | 1,729,920 | +380,704 | +28.22% |

## Per-run ranges

Min–max of the per-run median (ns/op); small timing differences can be measurement/layout noise, especially mapping syscalls.

| Metric | Disabled min–max | Enabled min–max |
|---|---:|---:|
| direct_compile_ns | 2494.000–2524.000 | 2448.000–2584.000 |
| direct_jit_ns | 2696.000–2782.000 | 2660.000–2820.000 |
| direct_cold_jit_ns | 9920.000–10956.000 | 10026.000–11378.000 |
| direct_call_ns | 0.810–0.890 | 0.800–0.900 |
| bc_v0_compile_ns | — | 114.000–128.000 |
| bc_v0_jit_ns | — | 252.000–278.000 |
| bc_v0_cold_jit_ns | — | 7478.000–8546.000 |
| bc_v0_call_ns | — | 0.810–0.870 |
| bc_v1_compile_ns | — | 206.000–238.000 |
| bc_v1_jit_ns | — | 348.000–374.000 |
| bc_v1_cold_jit_ns | — | 7766.000–8644.000 |
| bc_v1_call_ns | — | 0.790–0.840 |

This is a small-function microbenchmark, not an application throughput estimate. This arithmetic fixture uses direct lowering in both bytecode versions; v1 control flow, locals, deep stacks and extended operations still use the IR compiler.

Reproduce: `python3 scripts/bench_bc2jit_api.py --zig /path/to/zig --report bench/BC2JIT_API_RESULTS.md`.
Raw samples: `zig-out/bc2jit-compare/results.json`. This script does not rewrite regression baselines.

## Full-cycle acceptance

Threshold: 2.00×, against the assembler C API in the disabled build, with the same runtime cache in both builds.
- KXBC v0: median 10.30×; minimum paired run 9.78×
- KXBC v1: median 7.59×; minimum paired run 7.21×
Result: PASS.

## Validation

Correctness and formatting passed on the rebased source (base `32fe189`): native full tests with bytecode enabled/disabled, x86-64 API/runtime tests under Rosetta, LLVM-enabled full tests, and installed static/shared C consumers in both feature settings.

The three runtime-related baseline values were tightened using the median of three run medians: `jit_runtime_add_release` 173.3 ns, `e2e_aarch64_compile_run` 197.4 ns, `bc2jit_e2e_aarch64` 1048.5 ns. These gates passed on a subsequent run. Other baselines and tolerances were not changed.

The existing overall `bench-test` gate remains non-green for: `aarch64_emit`, `bc2jit_decode_64k`. The persistent `aarch64_emit` and `bc2jit_decode_64k` baseline failures were also reproduced on the original `20d0ed4` control before these API/runtime changes; transient unrelated thresholds vary with host load.
