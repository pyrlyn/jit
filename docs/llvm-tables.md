# Encoder tables from LLVM

`src/x86_64/gp_table.zig` (x86-64 legacy-encoded general-purpose instructions) is generated from
LLVM's X86 target description by a Zig tool, not from AsmJit. LLVM is a generation-time input
only; the generated table is committed and a normal build needs neither LLVM nor the generator.

| Table | Source | Status |
|---|---|---|
| `src/x86_64/gp_table.zig` | LLVM `X86.td` (llvmorg-23.1.4) via `tools/gen_tables.zig` | generated from LLVM |
| `src/x86_64/vec_table.zig` | AsmJit `db/isa_x86.json` via `scripts/gen_vec_tables.py` | not yet ported |
| `src/aarch64/a64_table.zig` | AsmJit `db/isa_aarch64.json` via `scripts/gen_a64_tables.py` | not yet ported |
| `src/x86_64/isa.zig`, `src/aarch64/isa.zig` | AsmJit DB via `scripts/gen_isa_tables.py` | not yet ported |

## Regenerating

```sh
git clone --depth 1 --filter=blob:none --sparse -b llvmorg-23.1.4 https://github.com/llvm/llvm-project
git -C llvm-project sparse-checkout set llvm/include llvm/lib/Target/X86
zig build gen-tables   -Dllvm-src=$PWD/llvm-project -Dllvm-tblgen=/path/to/llvm-tblgen   # writes the table
zig build check-tables -Dllvm-src=$PWD/llvm-project -Dllvm-tblgen=/path/to/llvm-tblgen   # committed == fresh
python3 scripts/gen_gp_tests.py   # optional: regenerate src/gp_tests.zig (expected bytes from clang)
```

Any `llvm-tblgen` that reads the pinned `.td` files works (tested with 23.1.4). `gen-tables` also
writes `scripts/.gp_forms.json` (git-ignored), read by `scripts/gen_gp_tests.py` and `scripts/fuzz_gp.py`.

## Generator (`tools/`)

- `llvm_json.zig`: reads `llvm-tblgen --dump-json` (records, dag operand lists, `{att|intel}` asm
  string variants, tied-operand constraints).
- `gen_x86_gp.zig`: for every `X86Inst` record with a legacy encoding (`OpEnc = EncNormal`, no
  pseudo, default address size, no PC-relative immediate), takes the Intel-syntax mnemonic and
  operands from `AsmString`, operand kinds from the operand classes (`GR32` -> `r32`, `i32mem` ->
  `m32`, `i32i8imm` -> `s8`, ...), prefixes from `OpSize`/`OpPrefix`, the opcode bytes from
  `OpMap` + `Opcode` (+ the fixed ModRM byte of `MRM_xx`/`MRMnX` forms), REX.W from `hasREX_W`,
  operand roles (ModRM.reg / ModRM.rm / opcode register) from `Form` and the MI operand order, the
  immediate size from `ImmT`, the CPU mode from `In64BitMode`/`Not64BitMode`. Condition-code
  records (`cmov${cond}`, `set${cond}`) are expanded into the 16 mnemonics.
- `gen_tables.zig`: entry point (`zig build gen-tables`).
- `compare_tables.zig`: form-by-form comparison of two tables (used for the migration below) and
  the `--exact` check behind `zig build check-tables`.

Project decisions that are not LLVM data live in `gen_x86_gp.zig` as small, commented lists:
which mnemonics this table offers (the rest is in `vec_table.zig`), the CPU feature per mnemonic
(LLVM's asm-level records rarely carry it), the reg,reg tie rule (MR before RM, RM for xchg),
`xchg` in both operand orders, the 16-bit-code-only variants that are left out, and a few manual
forms LLVM does not define (see below).

## Comparison with the previous (AsmJit-derived) table

`zig run tools/compare_tables.zig -- OLD NEW` against the table at `8d3de80`:

```
OLD: 201 mnemonics / 1065 forms, NEW: 201 / 1067; identical 114, reordered (ties kept) 85,
tie order differs 0, differing 2, only-OLD 0, only-NEW 0
```

- Every one of the 1065 old forms is present with the same operands, prefixes, opcode, REX.W,
  ModRM roles, immediate, features and mode, except `rsm` (below). 85 mnemonics list their forms
  in a different order; the encoder picks the shortest encoding and the first form on a tie, and
  the order of forms with the same operands and the same length is unchanged everywhere.
- `.mode = .x64` on REX.W forms is ignored by the comparison (REX.W forms are rejected in 32-bit
  mode by `gp.zig` regardless of the tag).
- Differences:
  - `lar r64, r32` / `lar r64, m16` (REX.W 0F 02 /r): new, valid encodings (LLVM `LAR64rr/rm`);
    the old table had them for `lsl` but not for `lar`.
  - `rsm` (0F AA): was 32-bit-mode only, now any mode (SDM: valid in 64-bit mode; LLVM has no
    mode restriction).
- Manual forms (not in LLVM 23): `mcommit` (AMD APM, F3 0F 01 FA), `ud0 r32, r/m32` (SDM,
  0F FF /r), `nop r/m, r` (0F 1F /r with a register field), `and r64/rax, imm32` (32-bit AND that
  zero-extends) and `mov r64, imm64` (= `movabs`).
- `src/gp_tests.zig` regenerated from the new forms: all 3126 previous cases are kept, 12 new
  (3138 total, bytes from clang); `scripts/gen_gp_tests.py` now skips 32-bit-only forms (they
  have no 64-bit LLVM spelling).

## Licensing

LLVM is Apache-2.0 WITH LLVM-exception. The generated table carries an attribution header and
`NOTICE` lists the derived file. AsmJit's license does not apply to `gp_table.zig` any more.
