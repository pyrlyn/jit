# Benchmark regression tests (`zig build bench-test`)

Zig std only, no new dependencies. Source: `bench/bench_test.zig`; baseline: `bench/baseline-<arch>-<os>.txt`
(currently `baseline-aarch64-macos.txt`, Apple Silicon Mac).

```sh
zig build bench-test                    # compare against the checked-in baseline, fail on regression
zig build bench-test -Dbench-tol=10     # override the tolerance for every case (percent)
zig build bench-update                  # re-measure and rewrite the baseline file, then commit it
```

The test binary and the library are always built `ReleaseFast`, whatever `-Doptimize` says.

## Cases (one test function each)

Only what exists on `main` is covered. The workloads come from the existing x86-64 codegen benchmark
(`bench/bench_x86_64.zig`, `bench/run.sh`, `scripts/bench_layers.py`, results in `bench/RESULTS.md`,
compared against AsmJit), plus the AArch64 / JitRuntime paths shown in `examples/example_aarch64.zig`.

| case | what is timed per iteration |
|---|---|
| `x86_64_straight` | reset + 256 units of straight-line mov/add/sub incl. imm64 (1569 instrs) + finalize |
| `x86_64_calls` | push/mov/call label/pop/call reg (1282 instrs) |
| `x86_64_loops` | 256 label loops: cmp/jcc/add/sub/jmp + bind (1281 instrs) |
| `x86_64_mem` | memory operands: SIB, lea, ALU/imul with mem, mov mem-imm (1537 instrs) |
| `x86_64_avx` | AVX/AVX2/FMA through the table-driven VEX encoder (1537 instrs) |
| `x86_64_avx512` | EVEX: opmask/zeroing, broadcast, {er}, compare into k, masked store (1537 instrs) |
| `aarch64_emit` | 256 copies of the sum loop (movz, labels, cbz, ldr, add/sub imm, b, movImm) + finalize |
| `jit_runtime_add_release` | `JitRuntime.add` (mmap + copy + protect) and `release` of a finalized 16-loop buffer |
| `e2e_aarch64_compile_run` | emit sum loop, finalize, `JitRuntime.add`, call it on 64 i64s, check result, release (native arm64 only) |
| `bc2jit_decode_small` | KXBC decode + validate of a ~100-byte module (4 all-opcode units), free |
| `bc2jit_decode_64k` | KXBC decode + validate of a 65 523-byte module (2730 all-opcode units, max u16 code length) |
| `bc2jit_lower_x86_64` | opcode -> x86-64 assembler translation of the 64 KiB module + finalize |
| `bc2jit_lower_aarch64` | same, AArch64 backend |
| `bc2jit_lower_riscv64` | same, RISC-V (RV64) backend |
| `bc2jit_stack_all_opcodes` | stack model: lower a 64-unit program that uses every opcode and walks depth 1..5 each unit, host backend |
| `bc2jit_e2e_aarch64` | decode + lower + `JitRuntime.add` + call + check vs reference interpreter + release (native arm64 only, 15%) |

The x86-64 scenarios are imported from `bench_x86_64.zig` (`Scenarios` is now `pub`), so the code
being timed is the same code the AsmJit golden-file tests check byte for byte.

bc2jit cases (`bc2jit_*`) import `bc2jit/lib.zig` as module `bc2jit`. The "all-opcode unit" is
`push_arg×3, dup, drop, mul, sub, push_i64, dup, mul, add, add` (24 bytes, every opcode, stack depth
1 -> 5 -> 1). IR / builder / register-allocator passes on `main` have no existing benchmark; add
cases here when one is wanted.

## Measurement and thresholds

- 200 warmup iterations, then 15 samples of 300 iterations; metric = **median ns per iteration**.
- Fails when `measured > baseline * (1 + tolerance)`. Tolerance: **5%**; **15%** for the two
  `JitRuntime` cases, which are dominated by mmap/mprotect syscalls and are much noisier.
- A failure prints the case, baseline, measured value and % delta:
  ```
  !!! BENCH REGRESSION: x86_64_loops
  !!!   baseline 8569.0 ns/iter, measured 9100.2 ns/iter, delta +6.2% (tolerance 5.0%)
  ```
- Noise handling: a case over tolerance is re-measured up to 3 times and the best median is kept, so
  one load spike doesn't fail the run, but a real regression still does. `bench-update` stores the
  median of 3 measurements. The baseline header records date, arch and load average.
- Baselines are per machine. Numbers from a loaded machine (this Mac often runs at load 10–20) are
  slower than `bench/RESULTS.md`; compare only against a baseline recorded on the same machine, and
  check `uptime` before trusting a failure.

## Standing rule: thresholds track the latest known-good baseline

When a change makes a case notably faster (more than ~5% below baseline, the test prints
`<-- IMPROVED`), run `zig build bench-update` and commit the new baseline in the same PR.
Stale, loose baselines would let later regressions through. Never loosen a baseline or tolerance
just to make a regressed case pass.
