# Compiler performance

Measured rounds of compile-time and peak-heap work on the IR compilers, newest
first. Every round uses the 12-case suite of `bench/compile_bench.zig` and requires
byte-identical machine code. Each table compares against its own baseline commit:
the 2026-11-21 baseline also contains #79, which has no section here, so its
"Before" columns are not the "After" columns of 2026-11-08.

## 2026-11-21

Baseline: `e90d5705e7d242a009976a033ecae4e5e7ec343c`. Host: Apple M3 Max,
arm64 macOS, Zig 0.17.1, ReleaseFast, C allocator.

The equal-weight suite result is **−5.6% time per compilation (1.06× throughput)**
and **−12.9% peak allocated heap**. The target of this round was to halve both
again; it was **not met**, and it cannot be met while the generated code stays
byte-identical (see "Limits").

5 alternating runs, 400 ms/case/run, seed 24301; all code checksums match.

| Backend/case | Before µs | After µs | Time reduction | Before peak KiB | After peak KiB | Peak reduction |
|---|---:|---:|---:|---:|---:|---:|
| aarch64/small | 1.833 | 1.833 | 0.0% | 1.9 | 1.5 | 21.1% |
| aarch64/medium | 28.625 | 28.083 | 1.9% | 20.3 | 20.3 | 0.0% |
| aarch64/large | 290.292 | 282.333 | 2.7% | 148.4 | 141.0 | 5.0% |
| aarch64/huge | 1475.875 | 1408.625 | 4.6% | 690.8 | 656.5 | 5.0% |
| x86_64/small | 5.166 | 5.333 | -3.2% | 3.8 | 3.2 | 15.8% |
| x86_64/medium | 81.666 | 75.125 | 8.0% | 30.3 | 29.7 | 2.0% |
| x86_64/large | 785.208 | 757.416 | 3.5% | 217.8 | 216.5 | 0.6% |
| x86_64/huge | 4164.792 | 4327.125 | -3.9% | 1015.5 | 1013.2 | 0.2% |
| riscv64/small | 1.042 | 1.083 | -3.9% | 3.1 | 1.2 | 61.3% |
| riscv64/medium | 10.791 | 10.667 | 1.1% | 13.0 | 12.3 | 5.4% |
| riscv64/large | 149.167 | 100.125 | 32.9% | 102.7 | 95.3 | 7.2% |
| riscv64/huge | 956.875 | 802.375 | 16.1% | 724.0 | 700.7 | 3.2% |

Equal-weight geometric mean across all 12 cases: time −5.6%, peak allocated heap −12.9%.
Compilation throughput: +5.9% (inverse of the time ratio).
These are suite averages, not a minimum improvement for each individual case. Peak heap excludes input IR, stack and allocator overhead/RSS.

The three negative time rows are run-to-run noise of about ±4% in this table; the
small cases are also only 25 to 130 ticks of the 41.67 ns clock. Measured one case
at a time (10 alternations each), the candidate/baseline time ratio is 0.975 to
0.984 for x86_64/huge, 1.008 to 1.016 for x86_64/small and 1.000 for
riscv64/small.

### Limits

- **Heap.** The machine code lives in the caller-owned CodeBuffer and is counted.
  It is 56 to 87% of the peak in the medium, large and huge cases: 632 of 701 KiB
  for riscv64/huge, 796 of 1013 KiB for x86_64/huge, and 516 KiB of code plus
  16 KiB of size-estimate slack out of 657 KiB for aarch64/huge. With identical
  code, only the remainder can shrink.
- **Time.** Emission is about 40% of a compilation. Every backend also walks the
  ops at least twice with a `switch` per op; on the random ops of the large and
  huge cases that branch mispredicts and costs about 5.6 ns per op (1 to 2 ns in
  the small and medium cases, where the predictor learns the repeated function).
  The walks that could still be removed are worth a few percent each.

### Changes

- The liveness fixpoint recomputes a block only when the live-in set of one of
  its successors changed since the block's last visit. This is the source of the
  riscv64/large and riscv64/huge time reductions.
- `ir.shapes` is a comptime table of operand offsets per op tag. Structural
  `verify`, `isIntOnly` and operand collection in `reg_classes` read operands
  through it instead of a `switch` per op. The call and terminator checks have
  one implementation (`checkSpecial`) shared with the op walk; the checks and the
  order of their errors are unchanged.
- `CodeBuffer.reserve` lets a backend reserve its labels and fixups exactly,
  which removes regrowth and capacity slack of both lists. RV64 counts labels
  and forward branches before lowering; AArch64 and x86-64 reserve their labels,
  and x86-64 reserves one fixup per jump that is left after relaxation.
- AArch64 per-op masks are `u8` instead of `u16`, and the call argument mask,
  which nothing read, is no longer built.

Left out on purpose: an 8 KiB stack scratch buffer is 5 to 14% faster in the small
cases but only moves memory from the heap to the stack, and a minimum code capacity
below 1 KiB would improve the small-case heap numbers without helping real callers.

### Correctness checks

- `zig build test`: 749 passed, 40 architecture-dependent skips; all 87 build steps pass.
- `zig build test -Dtarget=x86_64-macos` (Rosetta): 753 passed, 36 skips; all 87 steps pass.
- `zig build fmt-check` and `git diff --check` pass.
- All 72 generated-code checksums match the baseline: 12 cases for each seed
  `0x5eed`, `1`, `7`, `42`, `314159`, `0xdeadbeef`.
- A new test builds a function with every op kind, corrupts every `u32` field
  and argument list of every op in turn, misplaces terminators and breaks the
  function header, and requires the table-driven `verify` to return what the op
  walk returns. All seven structural errors occur in it. The shape table itself
  is checked against `Op.uses`/`Op.def` for a sample of each of the 17 op tags.

Reproduce with the commands under "Reproduction" below, with `BASELINE_SRC` in a
checkout of the baseline commit above.

## 2026-11-08

Baseline: `20d0ed4e4618d6ee040943644f5521db5fb91353`. Host: Apple M3 Max,
arm64 macOS, Zig 0.17.0, ReleaseFast, C allocator. All three backends emit code
on this host; this measures compilation, not execution of the generated code.

The equal-weight suite result is **−50.7% time per compilation (2.03× throughput)**
and **−71.1% peak allocated heap**, meeting the target of halving compile time and
peak heap on the 12-case suite. These are geometric means, not a guarantee for
every workload: individual time reductions range from 36.8% to 66.2%.

Input generation and checksumming are outside the timed region. Checksum mode uses
the same fixed call address (`0x100123450`) in both builds, so ASLR does not change
the generated bytes. No validation or compiler optimization option was disabled.

5 alternating runs, 400 ms/case/run, seed 24301; all code checksums match.

| Backend/case | Before µs | After µs | Time reduction | Before peak KiB | After peak KiB | Peak reduction |
|---|---:|---:|---:|---:|---:|---:|
| aarch64/small | 3.458 | 1.875 | 45.8% | 7.1 | 3.3 | 53.5% |
| aarch64/medium | 56.709 | 35.833 | 36.8% | 59.1 | 22.0 | 62.8% |
| aarch64/large | 537.167 | 337.917 | 37.1% | 446.3 | 147.7 | 66.9% |
| aarch64/huge | 3554.584 | 1852.458 | 47.9% | 2257.4 | 739.3 | 67.2% |
| x86_64/small | 15.834 | 5.416 | 65.8% | 17.5 | 4.0 | 77.1% |
| x86_64/medium | 233.333 | 78.958 | 66.2% | 171.0 | 34.9 | 79.6% |
| x86_64/large | 2338.542 | 845.416 | 63.8% | 1621.7 | 229.5 | 85.8% |
| x86_64/huge | 12193.125 | 5101.584 | 58.2% | 8153.6 | 1087.7 | 86.7% |
| riscv64/small | 1.875 | 1.083 | 42.2% | 6.5 | 3.1 | 52.3% |
| riscv64/medium | 20.125 | 10.792 | 46.4% | 44.5 | 14.8 | 66.7% |
| riscv64/large | 235.917 | 144.541 | 38.7% | 344.1 | 107.6 | 68.7% |
| riscv64/huge | 1801.417 | 992.417 | 44.9% | 1746.0 | 731.5 | 58.1% |

Equal-weight geometric mean across all 12 cases: time −50.7%, peak allocated heap −71.1%.
Compilation throughput: +102.9% (inverse of the time ratio).
These are suite averages, not a minimum improvement for each individual case. Peak heap excludes input IR, stack and allocator overhead/RSS.

Each compiler now uses a bounded **2 KiB stack scratch buffer**, with fallback to
the supplied allocator. Linear allocation also uses a **768-byte active-interval
array** (at most 64 physical registers). This is a heap metric, not a measurement
of total process memory or stack use; the change does not claim a 71.1% RSS saving.

### Changes

- Linear allocation without spill splitting visits IR operands directly while
  collecting liveness and local interval endpoints. It stores only block/edge
  descriptors and scans tags separately for calls. Graph allocation and splitting
  retain 8-byte compact instruction records with one sized operand allocation.
  The public slice-based allocator API remains unchanged.
- Type inference checks known operand types in the same instruction walk. Forward
  references retain one constraint per vreg, checked after copy/edge propagation;
  undefined values still default to i64. This removes a second full IR walk.
- Temporary compiler allocations use bounded stack scratch with heap fallback;
  the active register set is bounded by the 64-bit physical-register mask.
  Machine code stays in the caller-owned CodeBuffer.
- Liveness gen/kill sets and local interval endpoints are collected in one forward
  walk. Consecutive live-out runs extend intervals only at their endpoints; spill
  slot expiration skips scans until the earliest occupied slot can expire.
- Default x86-64 compilation encodes ordinary instructions directly and retains
  only branch/label events for relaxation and relocation. Dead-code and
  jump-to-next removal preserve the editable Builder's output. Public `lower`
  still targets the Builder, and disabling branch shortening retains its original
  serialization behavior.
- The editable x86-64 Builder's instruction payloads use alignment 1, removing
  padding from instruction/node storage. Zig tracks reduced alignment for accesses;
  the native in-memory layout changes, not the emitted instruction encoding.
- Builder insertion/emission keeps its fast path inline and growth out of line.
  Growth reserves 25% slack, and fresh code buffers start at 1 KiB instead of 4 KiB.
- The existing encoding-form validation is inlined, retaining its checks.

### Reproduction

Use the **same current** `bench/compile_bench.zig` harness for both library revisions.
`zig build bench-compile --verbose` prints the benchmark compiler command and its
`pyrlyn_jit_options` module path. Reuse that options file in both commands below.
Point `BASELINE_SRC` at `src/pyrlyn_jit.zig` in a checkout of the baseline commit:

```sh
# Set these to your baseline source and the options path from --verbose.
BASELINE_SRC=/path/to/baseline/src/pyrlyn_jit.zig
OPTIONS=.zig-cache/o/HASH/options.zig

zig build-exe -Ofast --dep pyrlyn_jit -Mroot=bench/compile_bench.zig \
  -Ofast --dep pyrlyn_jit_options -Mpyrlyn_jit="$BASELINE_SRC" \
  -Mpyrlyn_jit_options="$OPTIONS" -lc -femit-bin=/tmp/compile-before
zig build-exe -Ofast --dep pyrlyn_jit -Mroot=bench/compile_bench.zig \
  -Ofast --dep pyrlyn_jit_options -Mpyrlyn_jit=src/pyrlyn_jit.zig \
  -Mpyrlyn_jit_options="$OPTIONS" -lc -femit-bin=/tmp/compile-after
python3 scripts/compare_compile_bench.py /tmp/compile-before /tmp/compile-after \
  --repeats 5 --ms 400
```

The comparison script requires all 12 cases, rejects differing machine-code
checksums, alternates binary order and takes the median of each case's run medians.
It reports both individual results and the geometric mean of candidate/baseline
ratios. To inspect another deterministic workload, pass `--seed INTEGER`.

### Correctness checks

- `zig build test`: 261 passed, 17 architecture-dependent skips; all 75 build steps pass.
- `zig build test -Dtarget=x86_64-macos` (Rosetta): 265 passed, 13 skips; all 75 steps pass.
- `zig build fmt-check` and `git diff --check` pass.
- All 72 generated-code checksums match the baseline: 12 cases for each seed
  `0x5eed`, `1`, `7`, `42`, `314159`, `0xdeadbeef`.
- Eighty randomized branch/label programs produce byte-identical output and node
  counts through the direct compiler sink and editable Builder passes, including
  pre-existing code/fixups, dead instructions, comments, and short/long back edges.
  Exhaustive allocation-failure testing verifies scratch-storage cleanup.
- New randomized tests compare direct-IR, compact and slice-based allocation
  results for GP/FP classes, both algorithms, splitting, 16-argument calls and
  back edges to entry. Forward type constraints, undefined copy cycles, and stack
  scratch alignment/growth/fallback cleanup have focused regression coverage.
- The legacy `bench-test` reference interpreter needed an explicit rejection of
  non-v0 opcodes after KXBC added v1 variants; that build failure also reproduces
  on the baseline commit. Its benchmark workloads are unchanged.

#### Existing emitter benchmark thresholds

With the harness build repaired, `zig build bench-test` reports 16 passed and two
failed historical thresholds. A reference binary built with this same repaired
harness against `20d0ed4` fails the same two cases:

| Case | Historical threshold ns | Baseline commit measured ns | Candidate measured ns (two runs) |
|---|---:|---:|---:|
| aarch64_emit | 3530.6 | 4324.3 | 4320.4–4358.3 |
| bc2jit_decode_64k | 151739.2 | 218390.3 | 215707.4–230758.9 |

These thresholds were not loosened. After `zig build bench-update`, only the
four cases with clear improvements from this change were retained in the checked-in
baseline: `x86_64_straight`, `x86_64_calls`, `x86_64_loops`, `bc2jit_lower_x86_64`.
All other thresholds retain their previous values. The full emitter gate therefore
still exposes the two pre-existing failures rather than hiding them.
