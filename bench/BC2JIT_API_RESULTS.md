# bc2jit C API benchmark

Host: macOS-27.0.1-arm64-arm-64bit-Mach-O; Apple M3 Max. Zig 0.17.0.
ReleaseFast, default backends and hooks, validation off, unstripped artifacts. Identical source/options except `-Dbc2jit`.
7 runs per build, alternating order; median of per-run medians. Each run: 200 warmups, 15 samples × 500 iterations (cached calls: 100000).

## Cost of enabling support

Same assembler C API workload, no KXBC calls in the timed regions.

| Metric (ns/op) | Disabled | Enabled | Delta |
|---|---:|---:|---:|
| direct_compile_ns | 2518.000 | 2466.000 | -2.07% |
| direct_jit_ns | 8688.000 | 8496.000 | -2.21% |
| direct_call_ns | 0.810 | 0.840 | +3.70% |

## Assembler versus bytecode API

Both paths compile `add(a,b)` and check result 42. KXBC v0 emits byte-identical code to the assembler path (checked by the benchmark); v1 uses the IR compiler.
The assembler C API performs mnemonic/form lookup for each instruction; KXBC v0 uses specialized Zig emitters. These numbers do not compare KXBC with direct Zig assembler calls.
`compile`: allocate, emit/decode+lower, finalize, free; no executable mapping. `jit`: same plus map, one call, release. `call`: cached function only (includes loop, indirect call and volatile result store).

| Metric (ns/op) | Assembler | KXBC v0 | KXBC v1 |
|---|---:|---:|---:|
| compile | 2466.000 | 116.000 | 2552.000 |
| jit | 8496.000 | 5972.000 | 8556.000 |
| call | 0.840 | 0.830 | 1.660 |

Generated code: v0 20 bytes, v1 108 bytes.

## Artifact size

Sizes on disk in bytes; executable statically links the same C API benchmark in both builds. Disabled builds retain only bytecode API stubs; both binaries reference those symbols.

| Artifact | Disabled | Enabled | Delta bytes | Delta |
|---|---:|---:|---:|---:|
| bin/bench_bc2jit_api | 1,222,656 | 1,589,312 | +366,656 | +29.99% |
| bin/bench_x86_64 | 534,816 | 534,816 | +0 | +0.00% |
| bin/example_aarch64 | 401,216 | 401,216 | +0 | +0.00% |
| bin/example_hooks | 418,352 | 418,352 | +0 | +0.00% |
| bin/example_x86_64 | 417,184 | 417,184 | +0 | +0.00% |
| lib/libpyrlyn-jit.a | 1,737,168 | 4,485,920 | +2,748,752 | +158.23% |
| lib/libpyrlyn-jit.dylib | 1,348,944 | 1,724,176 | +375,232 | +27.82% |

## Per-run ranges

Min–max of the per-run median (ns/op); small timing differences can be measurement/layout noise, especially mapping syscalls.

| Metric | Disabled min–max | Enabled min–max |
|---|---:|---:|
| direct_compile_ns | 2482.000–2536.000 | 2426.000–2576.000 |
| direct_jit_ns | 8314.000–8774.000 | 8174.000–8806.000 |
| direct_call_ns | 0.790–0.830 | 0.810–0.940 |
| bc_v0_compile_ns | — | 112.000–124.000 |
| bc_v0_jit_ns | — | 5538.000–6236.000 |
| bc_v0_call_ns | — | 0.810–0.840 |
| bc_v1_compile_ns | — | 2448.000–2628.000 |
| bc_v1_jit_ns | — | 8202.000–9202.000 |
| bc_v1_call_ns | — | 1.620–1.710 |

This is a small-function microbenchmark, not an application throughput estimate. The v1 result includes fixed IR construction/register-allocation costs.

Reproduce: `python3 scripts/bench_bc2jit_api.py --zig /path/to/zig --report bench/BC2JIT_API_RESULTS.md`.
Raw samples: `zig-out/bc2jit-compare/results.json`. Existing regression baselines are unchanged.

## Validation and existing regression thresholds

Correctness checks passed: `zig build test` with bytecode support enabled and disabled,
Zig/C API tests with both optional backends disabled, and x86-64 API/CLI tests under Rosetta.
Installed static and shared libraries passed the C API tests using the system C compiler,
for both feature settings. Formatting and whitespace checks passed.

The existing `bench-test -Dbc2jit=false` passed. With support enabled, the checked-in
regression gate was not fully green: the repeat failed `aarch64_emit` (4394.2 ns versus
3530.6 baseline), `jit_runtime_add_release` (5763.9 versus 4935.1), and `bc2jit_decode_64k`
(218233.2 versus 151739.2). Machine load measured during this check was 42.91/21.75/16.76.

A control build from the original commit `20d0ed4`, with only the necessary exhaustive-switch
compile fix in the v0-only benchmark interpreter, reproduced the `aarch64_emit` (4336.3 ns)
and `bc2jit_decode_64k` (217306.8 ns) failures. Its runtime map/release case measured 5364.3 ns
and passed. These baseline failures and variable syscall measurements should not be treated
as evidence that the new API caused a slowdown; they also prevent claiming the old performance
gate is fully passing. No baseline or tolerance was changed.
