# bc2jit C API benchmark

Host: macOS-27.0.1-arm64-arm-64bit-Mach-O; Apple M3 Max. Zig 0.17.0.
ReleaseFast, default backends and hooks, validation off, unstripped artifacts. Identical source/options except `-Dbc2jit`.
7 runs per build, alternating order; median of per-run medians. Each run: 200 warmups, 15 samples × 500 iterations (cached calls: 100000).

## Cost of enabling support

Same assembler C API workload, no KXBC calls in the timed regions.

| Metric (ns/op) | Disabled | Enabled | Delta |
|---|---:|---:|---:|
| direct_compile_ns | 2514.000 | 2536.000 | +0.88% |
| direct_jit_ns | 2688.000 | 2696.000 | +0.30% |
| direct_call_ns | 0.840 | 0.840 | +0.00% |

## Assembler versus bytecode API

Both paths compile `add(a,b)` and check result 42. KXBC v0 emits byte-identical code to the assembler path (checked by the benchmark); v1 uses direct lowering for compatible straight-line code and the IR compiler otherwise.
The assembler C API performs mnemonic/form lookup for each instruction; KXBC v0 uses specialized Zig emitters. These numbers do not compare KXBC with direct Zig assembler calls.
`compile`: allocate, emit/decode+lower, finalize, free; no executable mapping. `jit`: same plus acquire an executable mapping, one call, release ownership to the bounded runtime cache. `call`: cached function only (includes loop, indirect call and volatile result store).

| Metric (ns/op) | Assembler | KXBC v0 | KXBC v1 |
|---|---:|---:|---:|
| compile | 2536.000 | 120.000 | 218.000 |
| jit | 2696.000 | 260.000 | 362.000 |
| cold_jit | 10412.000 | 7856.000 | 8298.000 |
| call | 0.840 | 0.840 | 0.840 |

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
| lib/libpyrlyn-jit.a | 1,742,352 | 4,883,048 | +3,140,696 | +180.26% |
| lib/libpyrlyn-jit.dylib | 1,349,216 | 1,729,920 | +380,704 | +28.22% |

## Per-run ranges

Min–max of the per-run median (ns/op); small timing differences can be measurement/layout noise, especially mapping syscalls.

| Metric | Disabled min–max | Enabled min–max |
|---|---:|---:|
| direct_compile_ns | 2508.000–2544.000 | 2522.000–2552.000 |
| direct_jit_ns | 2590.000–2698.000 | 2688.000–2726.000 |
| direct_cold_jit_ns | 10274.000–10668.000 | 10134.000–10812.000 |
| direct_call_ns | 0.840–0.840 | 0.840–0.840 |
| bc_v0_compile_ns | — | 118.000–122.000 |
| bc_v0_jit_ns | — | 254.000–266.000 |
| bc_v0_cold_jit_ns | — | 7722.000–8386.000 |
| bc_v0_call_ns | — | 0.810–0.840 |
| bc_v1_compile_ns | — | 212.000–242.000 |
| bc_v1_jit_ns | — | 342.000–380.000 |
| bc_v1_cold_jit_ns | — | 8052.000–8660.000 |
| bc_v1_call_ns | — | 0.810–0.840 |

This is a small-function microbenchmark, not an application throughput estimate. This arithmetic fixture uses direct lowering in both bytecode versions; v1 control flow, locals, deep stacks and extended operations still use the IR compiler.

Reproduce: `python3 scripts/bench_bc2jit_api.py --zig /path/to/zig --report bench/BC2JIT_API_RESULTS.md`.
Raw samples: `zig-out/bc2jit-compare/results.json`. Existing regression baselines are unchanged.

## Full-cycle acceptance

Threshold: 2.00×, against the assembler C API in the disabled build, with the same runtime cache in both builds.
- KXBC v0: median 10.34×; minimum paired run 9.96×
- KXBC v1: median 7.43×; minimum paired run 7.08×
Result: PASS.
