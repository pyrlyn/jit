# Security

ZigJIT generates and runs machine code, so memory-safety and W^X are treated as hard requirements.

## Executable memory (W^X)
- Memory is never writable and executable for the same thread at the same time.
- Linux / Intel macOS: code is copied into an RW mapping, which is then `mprotect`ed to RX before
  a pointer is returned. A failing `mprotect` unmaps the region and returns `ProtectFailed`.
- Apple Silicon: `MAP_JIT` + `pthread_jit_write_protect_np`. Writes happen inside a short
  per-thread window that is always closed again (`defer`), even on error paths.
- Dual mapping (`allocDual` / `addDual`): one RW view that is never executable and one RX view that
  is never writable. `DualFunction.write` is bounds-checked (`OutOfBounds`).
- Unused bytes of every code page are trap-filled (`int3` on x86-64, `udf #0` on AArch64).
- AArch64: the instruction cache is invalidated after every write (`sys_icache_invalidate` on
  macOS, `__clear_cache` elsewhere).

## Buffers, labels and relocations
- Code size is capped (`max_size`, at most 2 GiB); labels are capped (`max_labels`).
- `addFixup` rejects unknown labels; `finalize` checks that every patched field lies inside the
  code, every label is bound and every displacement fits its field. All checks run before
  anything is patched, so a failed `finalize` leaves the code unchanged.
- `labelOffset` returns `null` for foreign/unknown labels instead of reading out of bounds.

## Operands
- Encoders return errors (`NoMatchingForm`, `InvalidOperand`, ...) for operands no form accepts;
  out-of-range register ids, immediates and non-finite FP immediates are rejected, never encoded.
- Optional validation layer (`safety.Options`): operand checks, CPU-feature allow/deny lists,
  pre/post emission hooks.

## Testing
- `src/fuzz.zig`: robustness harnesses for the AArch64 encoder, the x86-64 GP encoder and the
  code buffer (labels/fixups). `zig build test` runs them over a fixed PRNG stream
  (350k cases); the same harnesses accept `std.testing.Smith` for `zig build test --fuzz`.
  Zig 0.16.0's test runner fails to compile in `--fuzz` mode (`std.debug.writeStackTrace`
  type mismatch in `lib/compiler/test_runner.zig`), so `zig build fuzz` runs the same harnesses
  from a standalone driver (`src/fuzz_main.zig`, ReleaseSafe) over fresh random seeds:
  `ZIGJIT_FUZZ_ROUNDS` (default 20, 0 = forever), `ZIGJIT_FUZZ_ITERS` (cases per harness per
  round, default 100000), `ZIGJIT_FUZZ_SEED` (replay a failing seed, printed on failure).
- Builds used for testing keep Zig's safety checks (Debug/ReleaseSafe): overflow and bounds
  violations panic instead of being undefined behaviour.

## Reporting
Please report vulnerabilities privately via GitHub Security Advisories on this repository
rather than in public issues.
