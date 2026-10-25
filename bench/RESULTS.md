# x86-64 codegen benchmark (Apple M3 Max, native arm64 host, 2026-10-25)
ns per emitted instruction; median of per-run averages / best single iteration; 9 runs x 5000 iterations.
AsmJit 0464e1c Release (-O3), `kOptimizeForSize`; ZigJIT ReleaseFast. All outputs byte-identical.

| Scenario | instrs | bytes (both) | AsmJit median / min | ZigJIT median / min | speedup (median) |
|---|--:|--:|--:|--:|--:|
| straight | 1569 | 6834 | 9.75 / 7.67 | 2.23 / 1.96 | 4.4x |
| calls | 1282 | 3330 | 10.45 / 8.61 | 2.51 / 2.15 | 4.2x |
| loops | 1281 | 4865 | 14.58 / 11.74 | 2.98 / 2.60 | 4.9x |
| mem | 1537 | 10112 | 10.90 / 9.14 | 2.35 / 2.06 | 4.6x |

## Optimizations (ZigJIT medians, ns/instr: straight / calls / loops / mem)
| Version | change | result |
|---|---|---|
| v0 | byte-by-byte `put8` through `*CodeBuffer` (`assembler_v0_baseline.zig.txt`) | 5.22 / 5.36 / 6.60 / 11.80 |
| v1 | assemble into a 16-byte stack array, commit with one 16-byte store | 5.11 / 2.99 / 3.31 / **18.45** (regression: store-forwarding stalls) |
| v2 | accumulator in two u64 registers instead of a stack array | **2.19 / 2.41 / 2.88 / 2.16** |

## Safety layer overhead (2026-10-25, Apple Silicon, ReleaseFast)

`python3 scripts/build_bench_bins.py && python3 scripts/bench_layers.py 15 5000`.
All binaries run interleaved in one session; median avg ns/instr. `baseline` = build before the
safety layer (`bench/pre-safety/`), `off` = `-Dvalidate=false -Dhooks=false`,
`validate` = `-Dvalidate=true`, `validate_hooks` = validate + `CountingHooks` attached.
The Mac was under background load (load average ~15), so absolute numbers differ from the
table above; compare columns within this table. Output is byte-identical to AsmJit in all
configurations (checked by `cmp` and by the harness tests).

| scenario | asmjit | baseline | off | validate | validate_hooks |
|---|--:|--:|--:|--:|--:|
| straight | 8.21 | 2.09 | 2.17 | 2.21 | 2.18 |
| calls | 9.16 | 1.29 | 1.37 | 1.72 | 1.71 |
| loops | 12.42 | 1.55 | 1.63 | 1.96 | 1.92 |
| mem | 9.79 | 2.17 | 2.17 | 2.49 | 2.13 |

Off-mode is within run-to-run noise of the baseline (in a quieter run it was equal or
slightly faster: 1.76/1.09/1.33/1.81 vs 1.87/1.15/1.36/1.81). Validation costs ~0–25%
(label-heavy scenarios pay for the label-id check and building `Ops`); a counting hook adds
nothing measurable on top. All configurations stay 4–7× faster than AsmJit.
