# TODO: remaining architectures and x86-64 work (plan only, no code yet)

Rules that every new backend follows (same as x86-64 and AArch64):
- its own module `src/<arch>/assembler.zig`, compiled only when listed in `-Dtargets=`;
- `AssemblerT(comptime Options)` with the same safety layer (`validate`, `Hooks`);
- no vtables or runtime dispatch on the emit path; host and target independent;
- byte-exact tests generated from the LLVM assembler (`scripts/gen_enc_tests.py`), execution
  tests only when host == target (or under an emulator such as Rosetta / qemu-user).

Builder, Compiler and register allocator: [docs/ROADMAP.md](docs/ROADMAP.md).

## 1. x86 (32-bit)
- Module `src/x86/`; share REX-less encoder pieces with `x86_64` through a comptime
  `mode: enum { x86, x86_64 }` parameter, not a runtime flag.
- Registers eax..edi only; no REX; `inc/dec r32` one-byte forms (0x40..0x4F) become legal.
- Addressing: 32-bit ModRM/SIB, absolute disp32 without RIP-relative.
- Calling conventions: cdecl, stdcall, fastcall (Win32), SysV i386.
- Tests: byte-exact vs `clang -target i386`; execution only on a 32-bit capable host (not macOS).
- Estimate: 1–2 weeks after the instruction-DB generator.

## 2. RISC-V (RV64GC first, RV32 later)
- Base RV64I + M + A, then F/D, then C (compressed, 16-bit) as an opt-in size mode.
- Fixed 32-bit words like AArch64; immediates split across fields (I/S/B/U/J types).
- `li` materialization (lui + addi, + slli chains for 64-bit), `auipc`+`jalr` for far calls.
- Labels: B-type ±4 KiB, J-type ±1 MiB, fixup kinds `rv_b12`, `rv_j20`, `rv_auipc_pair`.
- Validation: x0 as destination (writes discarded), immediate ranges, alignment with/without C.
- Tests: byte-exact vs `clang -target riscv64`; execution under qemu-user (optional CI job).
- Estimate: 2–3 weeks for RV64IMAFD+C subset.

## 3. MIPS (MIPS32/MIPS64 r2, later r6)
- Fixed 32-bit words; branch delay slots (pre-r6) must be explicit in the API or filled by
  the assembler with `nop` (configurable).
- `lui`/`ori` for constants, `jal`/`jalr`, 16-bit branch offsets (±128 KiB).
- Endianness is a target parameter (big/little) chosen at comptime.
- Tests: byte-exact vs `clang -target mips64el`; execution under qemu-user.
- Estimate: 2 weeks; low priority.

## 4. Others (only if needed)
- PowerPC64 LE, LoongArch64, WebAssembly (as a non-native target).
- AArch32/Thumb-2 (AsmJit has the DB only).

## 5. x86-64: 64-bit operands and immediates

### Done
- REX.W on all GP forms; REX.R/X/B for r8–r15 in reg, index and base fields.
- Full ModRM/SIB: rsp/r12 as base need SIB; rbp/r13 as base need disp8 0 (mod=00 means
  RIP/disp32); no-base SIB form `[index*scale + disp32]`; disp8 vs disp32 choice.
- Shortest `mov r64, imm` (same as AsmJit `kOptimizeForSize`):

| Value | Encoding | Size |
|---|---|---|
| 0 .. 0xFFFFFFFF | `mov r32, imm32` (upper half zeroed) | 5 B (6 B for r8–r15) |
| -0x80000000 .. -1 | `mov r/m64, imm32` (sign-extended) | 7 B |
| everything else | `movabs r64, imm64` | 10 B |

- ALU imm8 sign-extended form (`0x83`) when the value fits in i8, accumulator short form for rax.
- Tested on ~2000 byte-exact cases (all registers × forms, boundary immediates, all SIB combos).

### Remaining
1. Operand sizes 8/16/32: `0x66` prefix for 16-bit, no REX.W for 32-bit (implicitly zero-extends),
   8-bit opcodes (`0x88`, `0x80`...). Register enum per size (`Gp8/Gp16/Gp32/Gp64`) or a sized
   `Reg{ id, size }`. 8-bit: `spl/bpl/sil/dil` need an empty REX (0x40); `ah/ch/dh/bh`
   cannot be combined with any REX -> validation error.
2. RIP-relative addressing `[rip + disp32]` (mod=00, rm=101) with label targets (new fixup
   kind `rip32`, relative to instruction end, including trailing immediate size).
3. imm64 exists only in `movabs`; ALU ops take imm32 sign-extended. Add helpers that fall back
   to a temp register (caller-provided) when a 64-bit constant does not fit.
4. Extension rules and pitfalls, to be documented in the API and covered by tests:
   - `0x80000000` fits u32 but not i32: `mov r32` (zero-ext) is right, `add r64, imm32` is not
     encodable (it would become -2^31);
   - `0xFFFFFFFF` -> `mov r32` gives 0x00000000FFFFFFFF, while `mov r/m64, -1` gives all ones;
   - `-1` -> 7-byte `mov r/m64, imm32`; `or r64, -1` -> imm8 form.
5. Size policy table (opt-in `.mov_policy = .size | .asmjit_default`): AsmJit's default emits
   7-byte `mov r64, imm32` for positive values; pyrlyn-jit keeps the 5-byte form by default.
   `xor r32, r32` for zero only through an explicit `zeroR` (it clobbers flags).
6. Avoid redundant REX (already done for r/m without W; keep it when adding 32-bit forms).
7. More forms: shifts (`shl/shr/sar` by 1, imm8, cl), `neg/not`, `div/idiv` (rdx:rax), `movzx/movsx`,
   `cmovcc`, `setcc`, `xchg`, `bt*`, `cqo`.

### Tests
- exhaustive register × form byte-exact vs LLVM (extend the generator for each size);
- boundary immediates for each width: 0, 1, -1, 127, -128, 128, -129, 0x7FFF, 0x8000,
  0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 2^32, i64 min/max;
- all SIB base/index/scale × disp0/disp8/disp32, RIP-relative with forward/backward labels;
- execution round-trip under Rosetta (compute and compare with Zig results);
- fuzz differential: random instruction/operand streams, decode the bytes with `objdump` /
  LLVM MC disassembler and compare to the intended instruction.

Estimate: 2–3 weeks (sizes + RIP + extra forms), mostly mechanical once the generator exists.

## 6. x86-64 SIMD

### Order
1. SSE, SSE2 (baseline on x86-64), SSE3, SSSE3, SSE4.1, SSE4.2.
2. AVX (VEX, 128/256-bit, non-destructive 3-operand).
3. AVX2, FMA3, BMI1/2 (also VEX).
4. AVX-512 F / VL / BW / DQ (EVEX): mask registers k0–k7 (`{k}`, `{z}`), embedded broadcast
   `{1toN}`, embedded rounding / SAE, compressed `disp8*N`.
5. Maybe AVX10.x (converged AVX-512), APX later.

### Design
- Register types in the `x86_64` module: `Xmm` (0–31), `Ymm` (0–31), `Zmm` (0–31), `KReg` (0–7).
  xmm16–31 are EVEX-only -> validation error in SSE/VEX forms.
- Encoders:
  - legacy SSE: mandatory prefix (`66/F2/F3`) + REX + `0F`, `0F 38`, `0F 3A` maps;
  - VEX 2-byte (`C5`) when possible (map 0F, no X/B/W), else 3-byte (`C4`);
  - EVEX 4-byte (`62`) with R'/V'/aaa/z/L'L/b fields.
- Hot path: keep the `Enc` two-u64 accumulator. Max x86 instruction length is 15 bytes; EVEX
  (4) + opcode (1) + ModRM (1) + SIB (1) + disp32 (4) + imm8 (1) = 12, which fits the 16-byte
  store. Add a comptime assert that each encoder's worst case is <= 15.
- Comptime feature gating: `-Dx86-features=sse4_2,avx2,avx512f...` removes unsupported forms
  from the API at compile time (calling them is a compile error).
- Runtime: optional `CpuInfo` via CPUID; the validate layer can check the required feature of
  each instruction against a `CpuFeatures` set given in `Options`.
- Hooks: SIMD instructions get their own `Inst` ids and `Ops` fields (vector regs, mask,
  broadcast, rounding), so hooks see them like GP instructions.
- Ideally generate encoders and feature tables from AsmJit `db/isa_x86.json` with a
  `build.zig` step (Python or Zig generator), instead of writing ~1500 instructions by hand.

### Tests
- byte-exact per form vs LLVM (`clang -c` + `objdump`) and AsmJit;
- register-bank boundaries: xmm0–7, xmm8–15 (REX/VEX.R), xmm16–31 (EVEX only, rejected in SSE/VEX);
- memory: all SIB combos, EVEX disp8*N compression boundaries per vector length and element size;
- masks `{k1}`, `{k1}{z}`, k0 meaning "no mask", broadcast and rounding encodings;
- runtime execution gated on host CPU features (Rosetta has no AVX: those tests skip on Apple
  Silicon and run on x86 CI);
- validation rejects: missing feature, xmm16+ without EVEX, mask on non-maskable forms.

Estimate: SSE–SSE4.2 3 weeks, AVX/AVX2/FMA 2 weeks, AVX-512 4–6 weeks; with a DB generator
the total drops to ~6–8 weeks.

## 7. x86 30-bit immediates

x86 has no 30-bit immediate field: encodings carry imm8, imm16, imm32 or (only in `movabs`)
imm64. "30-bit" values matter as a *range*: constants that fit in 30 bits (|v| < 2^29 signed,
or < 2^30 unsigned), e.g. tagged small integers in JS/Lisp-style runtimes (31/30-bit SMIs),
offsets into a heap below 1 GiB, or AArch64/RISC-V-compatible constant ranges in a shared IR.

### Current state
- Any 30-bit value already fits imm32, so it is encoded as imm32 (or imm8 when it fits in i8).
- `movRI` picks 5 B `mov r32, imm32` for non-negative values and 7 B `mov r/m64, imm32` for
  negative ones; both cover the whole 30-bit range.

### Plan (text only)
1. Document the rule: on x86 a 30-bit constant never needs `movabs` or a temp register;
   sign-extension of imm32 is safe for signed 30-bit values, zero-extension via `mov r32` for unsigned.
2. Optional helpers for runtimes with tagged integers: `movSmi(r, v: i30)` / `addSmi`, which
   shift/tag at comptime and pick imm8/imm32; `i30`/`u30` parameter types make out-of-range
   values a compile error (Zig arbitrary-width integers).
3. Validation layer: when a caller passes an `i64` to a helper declared for 30-bit values,
   reject values outside the range with `error.InvalidImmediate`.
4. Cross-arch IR note: the Compiler can treat "fits in 30 bits" as a portable cheap-constant
   class (x86 imm32, AArch64 movz+movk ≤ 2 instructions, RISC-V lui+addi).

### Tests
- boundaries: 0, ±1, 127/−128, 2^29−1, −2^29, 2^30−1, 2^30 (must fall back to the general path),
  byte-exact vs LLVM for each form;
- execution round-trip under Rosetta for tagged-int helpers;
- validation rejects out-of-range values for 30-bit helpers.

Estimate: 2–3 days.
