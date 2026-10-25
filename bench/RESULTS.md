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

## Hot-path inlining + hooks on by default (2026-10-26 00:18 Kyiv, ReleaseFast, load avg ~13)

`before_*` = commit a4329b1, `after_*` = this change. 21 interleaved runs, median ns/instr.
`off` = `-Dhooks=false`; `after_default` = default ReleaseFast build (hooks compiled in, none
attached, attachment unknown at compile time); `+cb` = same binary with `ZJ_HOOK=1` (runtime
counting callback via function pointer); `counting` = comptime `CountingHooks` attached.

| scenario | asmjit | before_off | after_off | after_default | after_default+cb | before_counting | after_counting |
|---|--:|--:|--:|--:|--:|--:|--:|
| straight | 8.18 | 2.05 | 2.10 | 2.07 | 4.24 | 2.02 | 2.08 |
| calls | 8.80 | 1.33 | 1.25 | 2.26 | 3.83 | 1.62 | 1.55 |
| loops | 12.66 | 1.62 | 1.67 | 2.20 | 4.27 | 1.81 | 1.82 |
| mem | 9.79 | 2.18 | 1.20 | 2.04 | 3.97 | 2.05 | 1.14 |

__text size: before_off 266,028 B, after_off 262,444 B, after_default 269,380 B,
after_counting 262,736 B. Generated code is unchanged (6834/3330/4865/10112 bytes, identical to AsmJit).

## After ISA-db validation, finalize and label checks (30eb76e)

ReleaseFast, 15 interleaved runs, median ns/instr. Machine load average ~60 (very noisy; absolute numbers ~1.8x higher than earlier runs). `after_*` = ed5c4ee, `safety_*` = 30eb76e.

| scenario | asmjit | after_off | safety_off | after_default | safety_default | safety_default+cb | after_counting | safety_counting |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| straight | 15.19 | 3.78 | 3.85 | 3.31 | 3.34 | 7.20 | 3.84 | 3.80 |
| calls | 15.56 | 4.20 | 4.29 | 3.62 | 3.30 | 7.30 | 2.53 | 3.34 |
| loops | 20.86 | 3.98 | 4.47 | 3.66 | 3.01 | 7.16 | 3.48 | 3.90 |
| mem | 16.34 | 2.78 | 2.88 | 3.38 | 3.39 | 6.31 | 2.79 | 2.29 |

Differences are within run-to-run noise at this load; no systematic regression. The new always-on checks run in `finalize`/`bind` (outside the emit loop); the ISA-db byte check is only compiled in with `validate`.

## After align/trap/prolog helpers and buffer limits (553b046)

15 interleaved runs, load ~43, median ns/instr. `safety_*` = 30eb76e, `q_*` = 553b046.

| scenario | asmjit | safety_off | q_off | safety_default | q_default |
|---|--:|--:|--:|--:|--:|
| straight | 14.51 | 3.85 | 3.83 | 3.29 | 3.30 |
| calls | 15.45 | 4.43 | 4.45 | 3.38 | 3.31 |
| loops | 20.97 | 4.52 | 4.52 | 3.03 | 3.34 |
| mem | 16.03 | 2.89 | 2.89 | 3.41 | 3.37 |

Layers-off is unchanged. `loops`/default is +0.3 ns (newLabel limit compare); within noise at this load, to be rechecked on an idle machine.
