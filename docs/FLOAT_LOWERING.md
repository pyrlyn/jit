# Float lowering in the Compilers

Roadmap §2 ("lower float/vector IR"): scalar float IR (`src/core/ir.zig`) compiled by the
register-allocating Compilers.

## Status

| IR | x86-64 | AArch64 | x86 (32-bit) | RV64 |
|---|---|---|---|---|
| `f32`: `fconst`, `add/sub/mul/div/min/max`, `fcmp`, `load/store.f32`, `copy` | yes | yes | rejected | rejected |
| `f64` | not yet | not yet | rejected | rejected |
| `cvt` (i64 <-> f32/f64, f32 <-> f64) | not yet | not yet | rejected | rejected |
| vectors (`v4i32`, `v2i64`, `v4f32`, `v2f64`), `splat`, `extract` | rejected | rejected | rejected | rejected |

Rejected IR returns `error.UnsupportedType` (`PJIT_ERR_UNSUPPORTED` through the C API); it
is never miscompiled. The supported set is `reg_classes.support`.

## Design

- **Register classes** (`src/core/reg_classes.zig`): `ir.Func.typeCheck` gives every vreg a
  type; `i64` values are GP class, `f32`/`f64` FP class. Each class runs the existing linear-scan
  allocator separately over the same instruction positions, so call sites, positions and spill
  slots line up. FP spill slots follow the GP ones in the frame.
- **ABI**: IR parameters, call arguments and `ret` are `i64`, so float values never cross the
  ABI in registers. The FP run has no callee-saved registers: a float live across a call is
  spilled (8-byte slot).
- **x86-64**: SSE/SSE2 scalar forms (`addss`, `minss`, `ucomiss`, `movss`, `movq`, ...) as
  Builder nodes (`sseRR`, `sseRM`, `sseMR`, `movqXR`, ...), encoded by the shared vector table.
  xmm0..13 allocatable, xmm14/15 scratch (SysV); Win64 uses xmm0..3 and xmm4/5 only, because
  xmm6..15 are callee-saved there. `fcmp` uses `ucomis` + `setcc` (operands swapped for
  `lt`/`le`, PF combined for `eq`/`ne`) so NaN gives IR semantics.
- **AArch64**: scalar FP instructions (`fadd`, `fcmp`, `fcsel`, `fmov`, `ldr/str s|d`, ...),
  encodings checked against LLVM in a test. v0..7 and v16..29 allocatable (v8..15 are
  callee-saved in AAPCS64 and unused), v30/v31 scratch.
- **Semantics** (same as the interpreter): `min`/`max` return the second operand when
  unordered or equal (x86 `minss`; AArch64 lowers to `fcmp` + `fcsel`, not `fmin`); `fcmp` is
  ordered (false on NaN) except `ne`.

## Tests (`src/fp_tests.zig`)

- every op and every `fcmp` condition on all pairs of special values (±0, ±inf, NaN,
  subnormal, max, min-normal, ...), bit-exact against `ir.interpret`, with the normal register
  set and with one FP register (spill stress);
- loops and branches on `fcmp`, float values live across a call that clobbers FP registers,
  checked against the same loop in Zig;
- memory displacements (negative, 4-byte, large);
- unsupported float IR is rejected;
- random-program differential fuzz (300 programs per type);
- both compilers lower float IR on any host (bytes only); AArch64 encodings vs LLVM.
