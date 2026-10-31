# API

## Core
| Item | What it does |
|---|---|
| `CodeBuffer.init(gpa)` / `deinit()` | growable code buffer with labels and fixups |
| `cb.reset()` | drop code and labels, keep capacity (cheap reuse) |
| `cb.finalize()` | patch forward branches; returns `error.UnboundLabel` / `DisplacementOutOfRange` |
| `cb.code()` | the emitted bytes |
| `a.newLabel()`, `a.bind(label)` | labels; backward branches resolve at once, forward ones in `finalize` |
| `JitRuntime.add(&cb)` | map code executable (W^X; `MAP_JIT` on Apple arm64), `.ptr(T)`, `.release()` |

## x86-64 (`pjit.x86_64`)
Registers `Gp` (`.rax`..`.r15`), memory `ptr(base, disp)`, `ptrIdx(base, index, scale, disp)`,
or `Mem{ .index = .rcx, .scale = .x8, .disp = 16 }` (no base). All operands are 64-bit.

| Method | Instruction |
|---|---|
| `movRR`, `movRM`, `movMR`, `movMI` | `mov r,r` / `r,[m]` / `[m],r` / `qword [m],imm32` |
| `movRI(r, i64)` | shortest form: `mov r32,imm32` (5–6 B), `mov r/m64,imm32` (7 B), `movabs` (10 B) |
| `movabs(r, u64)` | always 10 B (patchable constant) |
| `lea` | `lea r,[m]` |
| `aluRR/RM/MR/RI(op, ...)`, `addRR`, `addRI`, `subRR`, `subRI`, `cmpRR`, `cmpRI`, `xorRR` | `add or adc sbb and sub xor cmp`; imm8 / imm32 / accumulator short form |
| `testRR`, `imulRR`, `imulRM`, `imulRRI` | `test`, `imul` |
| `push`, `pop`, `ret`, `nop` | |
| `call(label)`, `callR(r)`, `callAbs(addr)`, `jmp(label)`, `jmpR(r)`, `jcc(cond, label)` | rel8 for near backward targets, else rel32 |

## AArch64 (`pjit.aarch64`)
Registers `X` (`.x0`..`.x30`, `.zr_sp` = register 31, SP or XZR depending on the instruction).

| Method | Instruction |
|---|---|
| `movz`, `movk`, `movImm(rd, u64)` | 16-bit moves; `movImm` = movz + movk for non-zero halves |
| `mov`, `add`, `sub`, `mul`, `cmp` | register forms |
| `addImm`, `subImm`, `cmpImm` | 12-bit immediate forms (SP allowed) |
| `ldr`, `str` | `[xn, #off]`, `off` multiple of 8, < 32768 |
| `b`, `bl`, `bcond`, `cbz`, `cbnz`, `blr`, `ret`, `word(u32)` | branches; `word` emits a raw instruction |

## C / C++ / Rust (`include/pyrlyn_jit.h`)

A stable C ABI with a header-only C++ RAII wrapper (`include/pyrlyn_jit.hpp`). It is used from
Rust via plain `extern "C"`. Full reference: [docs/c-api.md](c-api.md).

```c
pjit_buffer *b; pjit_func *f;
pjit_buffer_new(&b);
pjit_a64_op ops[] = { pjit_a64_x(0), pjit_a64_x(0), pjit_a64_x(1) };
pjit_a64_emit(b, "add", ops, 3);          /* any AsmJit mnemonic; errors are return codes */
pjit_a64_emit(b, "ret", NULL, 0);
pjit_finalize(b);
pjit_jit_add(b, &f);                      /* W^X executable copy */
long r = ((long (*)(long, long))pjit_func_ptr(f))(40, 2);
```

Examples for every language live in `examples/{c,cpp,rust,zig}`. Run them with `zig build examples`
(add `-Drust=true` for Rust). They also run as part of `zig build test`.

