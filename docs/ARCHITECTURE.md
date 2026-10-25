# ZigJIT architecture vs AsmJit

Reference: AsmJit `0464e1c` (2026-10-16), read-only clone in `ref/asmjit` (~115k lines C++).
ZigJIT: ~1.3k lines Zig 0.16, stdlib only.

## AsmJit, in brief
| Layer | AsmJit | Notes |
|---|---|---|
| Container | `CodeHolder` | sections, `LabelEntry`s, `Fixup`s, `RelocEntry`s, address tables, attached emitters |
| Bytes | `CodeBuffer` (per section) | growable, arena-backed |
| Emitter API | `BaseEmitter` -> `BaseAssembler` / `BaseBuilder` / `BaseCompiler` | virtual `_emit(InstId, ops...)`; generated typed wrappers (`x86_emitter.h`, 4.4k lines) |
| Assembler | `x86::Assembler`, `a64::Assembler` | encodes immediately from instruction DB tables |
| Builder | node list (InstNode, LabelNode, Align, Embed, ConstPool) + passes | serializes to an Assembler |
| Compiler | Builder + virtual regs + `FuncNode` + RA pass (`ra_*`, liveness, CFG, spill/stack) | per-arch `x86_ra_pass`, `a64_ra_pass` |
| Instruction DB | `db/isa_x86.json`, `isa_aarch64.json` + `tools/tablegen-*.js` | generates `x86_inst_db.cpp` (1641 instr), `a64_inst_db.cpp` (796 instr), validation/RW info, CPU features |
| Runtime | `JitRuntime` -> `JitAllocator` (pooled, bitmapped blocks) -> `VirtMem` (MAP_JIT, dual mapping, W^X, icache flush) | |
| Misc | `Environment`/`Arch`, `CpuInfo`, `Formatter`/`Logger`, `ConstPool`, `FuncFrame`/calling conventions, `ErrorHandler`, PAuth, `ujit` (portable SIMD compiler) | |
| Backends | x86/x64, AArch64 (AArch32 DB only) | |

## ZigJIT mapping
| AsmJit | ZigJIT | Status |
|---|---|---|
| `CodeHolder` + `CodeBuffer` (single .text) | `core/code_buffer.zig: CodeBuffer` (bytes, labels, fixups, `reset`, `finalize`) | done (one section) |
| `Label`, `LabelEntry`, `Fixup` | `Label{id}`, `labels: ArrayList(u32)`, `Fixup{at,base,label,kind}` resolved in `finalize` | done: rel8/rel32 (x86), imm26/imm19 (a64) |
| `RelocEntry`, base address, cross-section | — | todo |
| `Environment`/`Arch` | `core/arch.zig` (host vs target kept separate) | done |
| `BaseEmitter` virtual `_emit` | **none on purpose**: user calls `zigjit.x86_64.Assembler` methods; everything comptime, no vtable | by design |
| `x86::Assembler` | `x86_64/assembler.zig` | subset (below) |
| `a64::Assembler` | `aarch64/assembler.zig` | small subset |
| x86 (32-bit) | — | planned as its own module `x86/` (shares encoder pieces with x86_64 via comptime) |
| Builder / Compiler / RA | — | todo (largest piece) |
| Instruction DB + tablegen | hand-written encoders | todo: generate Zig encoders at build time from AsmJit's `db/*.json` (a `build.zig` step) |
| `JitRuntime`/`VirtMem` | `core/jit_runtime.zig` (MAP_JIT + `pthread_jit_write_protect_np` on Apple arm64; RW→RX `mprotect` elsewhere) | basic + opt-in dual mapping (`addDual`: mach_vm_remap / memfd); no pooled allocator |
| `BaseEmitter` validation (`DiagnosticOptions::kValidateAssembler`) | `Options.validate` (comptime) | done for the current subset |
| `Logger` / emitter hooks | `Options.Hooks` (comptime type, `pre`/`post`) + `LogHooks`, `CountingHooks`, `CallbackHooks` | done |
| `CpuInfo`, `Formatter`, `ConstPool`, `FuncFrame`, `ErrorHandler`, PAuth, `ujit` | — | todo |

### Build-time selection
`zig build -Dtargets=x86_64,aarch64` (default both). Disabled backends are `void` in
`zigjit.zig`, so they are never analyzed or compiled. Hot path has no runtime dispatch:
`Assembler` is a one-pointer struct, each method is a direct encoder that writes one
instruction with a single 16-byte store into the pre-reserved buffer.

### x86-64 subset
mov (r,r / r,imm shortest / r,m / m,r / m,imm32), movabs, lea, add/or/adc/sbb/and/sub/xor/cmp
(r,r / r,imm8|imm32 incl. accumulator form / r,m / m,r), test, imul (r,r / r,m / r,r,imm),
push/pop, call (label / reg / absolute via r11), jmp (label / reg), jcc (16 conditions), ret, nop.
Memory: base, index*scale, disp8/disp32, no-base SIB, rsp/r12/rbp/r13 special cases. 64-bit operand size only.

### AArch64 subset
movz/movk/movImm (any u64), mov reg, add/sub (reg, imm12), mul, cmp (reg, imm12),
ldr/str (unsigned scaled offset), b/bl/b.cond/cbz/cbnz to labels, blr, ret.

## Remaining to reach AsmJit parity (realistic estimate, one experienced engineer)
| Feature | Effort |
|---|---|
| Instruction-DB-driven x86 encoder (all GP, x87, SSE..AVX-512, APX, VEX/EVEX, operand-size variants) generated from `isa_x86.json` | 6–10 weeks |
| Same for AArch64 (incl. NEON/SVE operands, logical immediates, shifted/extended regs) | 4–6 weeks |
| x86 32-bit module | 1–2 weeks (after DB generator) |
| Sections, relocations, base address, `ConstPool`, align/embed | 1–2 weeks |
| JitAllocator (pooled, bitmap, dual-mapping, W^X per thread) | 1–2 weeks |
| Builder (node list + passes) | 2 weeks |
| Compiler: virtual regs, calling conventions, FuncFrame, prolog/epilog | 3–4 weeks |
| Register allocator (CFG, liveness, local + global RA, spills) — AsmJit's hardest part | 6–10 weeks |
| Formatter, full DB-driven validation, CpuInfo | 2–3 weeks |
| Test suite parity (AsmJit's encoding tests, fuzzing vs llvm-mc) | ongoing, 2–3 weeks |
| **Total** | **~7–11 engineer-months** for Assembler+Builder+Compiler parity on x86-64 + AArch64 |

## Safety layer (validation + hooks)

Files: `src/core/safety.zig` (Options, event types, ready-made hooks), and the `AssemblerT`
generic in each backend.

### How it works
Each backend defines `AssemblerT(comptime opts: Options) type`. Every public emitter is a
one-line wrapper:

```zig
pub fn movRR(a: Self, dst: Gp, src: Gp) Error!void {
    if (comptime !checked) return a.rawMovRR(dst, src); // both layers off
    return a.emit(.mov_rr, .{ .r0 = dst, .r1 = src }, rawMovRR, .{ dst, src });
}
```

The `checked` shortcut matters: an earlier version always went through `emit` and relied on
LLVM to drop the unused `Ops` value. It did not always do so (label-heavy code was ~25%
slower in the calls/loops benchmarks), so the off path now never builds `Ops` at all.

When a layer is on, `emit` (inline) does:
1. `if (comptime opts.validate) try a.validate(id, ops);`
2. if `Hooks == void` (comptime) it tail-calls the raw encoder with `@call(.always_inline, ...)`;
3. otherwise, if a hook instance is attached: `pre` (may veto), raw encode, `post` with the
   offset and the encoded bytes.

With both layers off the wrapper is the raw encoder, and the struct holds only the buffer pointer. This is checked by a test (`@sizeOf`) and by
the benchmark (no slowdown, see bench/RESULTS.md).

AsmJit does the same job at runtime: `BaseEmitter::_emit` always goes through a virtual call,
checks `_diagnosticOptions` for validation, and calls the `Logger` if attached. ZigJIT moves
that choice to compile time.

### What is validated and why
The typed API already rules out most bad operands: `Gp`/`X` enums cannot hold an invalid
register, immediates have exact widths (`i32`, `u12`, `u16`, `u15`, comptime shift). The
validator covers what types cannot express: x86 `rsp` as SIB index, scale without index,
foreign labels; AArch64 register-31 ambiguity (SP vs XZR) and backward branch ranges.

### Test
`zig build test` runs `src/tests.zig` (validation, hooks, disabled = identical bytes) and the
generated byte-exact suite `src/enc_tests.zig`.

### Limitations
No CPU-feature checks (no SIMD yet), no RW/register-class info from an instruction DB;
`ev.bytes` of forward branches has the displacement unpatched until `finalize`.

## x86-64 emit hot path: inlining

Measured on the benchmark, not by rule:
- Forced `inline fn` (Zig semantic inlining) for the public emitters and raw encoders of
  `mov*`, `movabs`, `lea`, `aluRM/aluMR`, `test`, `imul*`, `callAbs`, the label helpers
  (`newLabel`, `bind`) and `CodeBuffer.newLabel/bind/addFixup` fast paths.
- Left to LLVM (`pub fn`, inlined by LLVM in practice): `aluRR`, `aluRI` (and `add/sub/cmp/xor`
  aliases), `push`, `pop`, `ret`, `nop`, `callR`, `jmpR`, `jmp`, `call`, `jcc`. Forcing these
  made the calls/loops scenarios 10–15% slower (bigger code, worse register allocation).
- Growth paths are `noinline` + `@branchHint(.cold)`: `CodeBuffer.grow`, `growLabels`, `growFixups`.
- The bench timing loop has no calls into emit functions (checked with `objdump -d`); the
  remaining calls are `grow*` (cold), timing and printing.
- With hooks compiled in, the hooked branch is `@branchHint(.unlikely)`.

## ISA database validation and always-on checks

- `scripts/gen_isa_tables.py` reads `ref/asmjit/db/isa_x86.json` / `isa_aarch64.json` and generates `src/x86_64/isa.zig` and `src/aarch64/isa.zig`: per implemented instruction, the allowed forms (x86: opcode, /digit, immediate size, REX.W; AArch64: mask/value, register-31 meaning, immediate width/scale) and the required CPU features. `A64_OP_OVERRIDE` corrects two DB bugs (`cbz Xn`, `ret Xn`).
- With `validate`, `encode()` matches the emitted bytes against the forms; on mismatch the bytes are rolled back and `error.EncodingMismatch` is returned.
- `Options.cpu_check`: `.none` (default), `.target` (`target_features` resolved at comptime; an instruction needing a missing feature is a compile error), `.host` (`cpu.host()` via CPUID+XGETBV on x86-64, `sysctlbyname hw.optional.arm.FEAT_*` on macOS arm64; `error.UnsupportedCpuFeature`).
- `CodeBuffer.finalize` always runs a check pass before patching: unbound referenced labels (`UnboundLabel`, list via `unresolvedLabels`), rel8/rel32/imm19/imm26 range (`DisplacementOutOfRange`), size > `maxInt(i32)` (`CodeTooLarge`, also from `grow`). On error nothing is patched. `bind` rejects unknown ids (`InvalidLabel`) and double binding (`LabelAlreadyBound`); branches always check label validity.

## Generated GP encoder (`src/x86_64/gp.zig`)

- The table holds forms with operand kinds (`r8..r64`, `m8..m64`, `m`, immediate kinds `i8/s8/u8/i16/u16/i32/s32/u32/i64`, fixed `al/ax/eax/rax/cl/dx/1`). Each form also has prefixes, REX.W, opcode bytes, `+r`, ModRM `/r` or `/digit`, immediate size, which operands go to ModRM.reg/rm/opcode, and CPU features.
- `gp.emit` matches every form of the comptime-known mnemonic, encodes each match into a 16-byte scratch and keeps the shortest. Immediates are normalized to the operand size, so `0xFFFFFFFF` on a 32-bit operand can use the imm8 form. 8-bit `spl/bpl/sil/dil` get an empty REX.
- RIP-relative is ModRM mod=00 rm=101 + disp32. With a label, a `rel32` fixup is added with `base` = end of instruction, so a trailing immediate is handled correctly.
- This path is separate from the hand-written hot-path emitters (whose bytes and benchmarks are unchanged). It is meant for coverage; the hot path stays specialized.
