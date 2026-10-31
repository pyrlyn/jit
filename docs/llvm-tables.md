# Encoder tables from LLVM

`src/x86_64/gp_table.zig` (x86-64 legacy-encoded general-purpose instructions) and the validation
tables `src/x86_64/isa.zig` and `src/aarch64/isa.zig` are generated from LLVM's X86 and AArch64
target descriptions by a Zig tool, not from AsmJit. LLVM is a generation-time input
only; the generated table is committed and a normal build needs neither LLVM nor the generator.

| Table | Source | Status |
|---|---|---|
| `src/x86_64/gp_table.zig` | LLVM `X86.td` (llvmorg-23.1.4) via `tools/gen_tables.zig` | generated from LLVM |
| `src/x86_64/vec_table.zig` | AsmJit `db/isa_x86.json` via `scripts/gen_vec_tables.py` | not yet ported |
| `src/aarch64/a64_table.zig` | AsmJit `db/isa_aarch64.json` via `scripts/gen_a64_tables.py` | not yet ported |
| `src/x86_64/isa.zig`, `src/aarch64/isa.zig` | LLVM `X86.td` / `AArch64.td` via `tools/gen_tables.zig` | generated from LLVM |

## Regenerating

```sh
git clone --depth 1 --filter=blob:none --sparse -b llvmorg-23.1.4 https://github.com/llvm/llvm-project
git -C llvm-project sparse-checkout set llvm/include llvm/lib/Target/X86 llvm/lib/Target/AArch64
zig build gen-tables   -Dllvm-src=$PWD/llvm-project -Dllvm-tblgen=/path/to/llvm-tblgen   # writes the tables
zig build check-tables -Dllvm-src=$PWD/llvm-project -Dllvm-tblgen=/path/to/llvm-tblgen   # committed == fresh (byte-exact)
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
- `gen_isa.zig`: `src/x86_64/isa.zig` and `src/aarch64/isa.zig` (see below).
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

## Validation tables (`isa.zig`)

`validate = true` checks every emitted instruction of the hand-written encoders against these
tables. `tools/gen_isa.zig` reads, per encoder entry, the LLVM record that defines it:

- x86-64: opcode bytes (`OpMap` + `Opcode`; the condition code is added for `AddCCFrm` records,
  `JCC_1`/`JCC_4`), opcode-register form (`AddRegFrm`), ModRM use and `/digit` (`Form`), immediate
  kind (`ImmT`), required features (`Predicates` naming a `SubtargetFeature`). `.llvm` is the
  record name, `.syntax` its Intel-syntax assembly string.
- AArch64: fixed bits from `Inst`; for an alias (`mov`, `mul`, `mneg`, `neg`, `cmp`, `cmp_imm`,
  `cset`, `lsr_imm`, `asr_imm`) the fixed operands of the LLVM `InstAlias` (XZR -> 31, literals) are
  applied to the fields of the underlying record. Rd/Rn accept SP when the operand class is
  `GPR64sp`; immediate width/scale come from the operand class (`a64_imm_classes`).
- `Feature` enums: the public names are unchanged (they are the API of `target_features` /
  `deny_features`, and `gp_table.zig`/`vec_table.zig` use them). Each name maps to an LLVM
  `SubtargetFeature` record (the generator fails if one is missing); names with no LLVM feature
  (x86: 30, e.g. `MSR`, `RDTSC`, `VMX`, `3DNOW`; AArch64: 3, `DGH`, `RPRFM`, `TME`) are kept with
  the reason next to them in `x86_features` / `a64_features`.

Compared with the previous AsmJit-derived tables (`src/isa_tests.zig` checks this): every x86-64
entry has the same opcode, `/digit`, immediate and features; every AArch64 entry has the same
fixed bits, SP rules, immediate width/scale and features, except `add_imm`, `sub_imm` and
`cmp_imm`: in LLVM the `addsub_shifted_imm64` operand owns bits 23:22 (shift), so bit 23 is no
longer a fixed bit (mask `0xFF000000` instead of `0xFF800000`). The encoders still emit 0 there;
a test checks it. `cbz` and `ret` no longer need the fix-ups the AsmJit database needed.

## Plan for the remaining tables

| Table | LLVM source | Measured gap (vs the AsmJit-derived table) | Handling |
|---|---|---|---|
| `vec_table.zig` (7131 forms) | `X86.td`, VEX/EVEX/XOP records (`OpEnc` = `EncVEX`/`EncEVEX`/`EncXOP`) and SSE/MMX legacy records | LLVM match for 6539; missing: 3DNow! (removed from LLVM), some SEV/TDX/SVM system instructions, AVX10.2 forms under their pre-rename names | Overrides file authored by us (`tools/x86_vec_overrides.zig`, SDM/APM references) for SEV/TDX/SVM; old AVX10.2 names kept as aliases of the renamed LLVM records; 3DNow! dropped (no current CPU, removed upstream) unless a user needs it. |
| | | AsmJit-only flags `lowprio`, `late_vex`, `er`, `sae` | `er`/`sae` from `hasEVEX_RC` / `hasEVEX_B` + operand class; `lowprio`/`late_vex` are encoder-selection rules: recompute from LLVM (e.g. prefer the shorter VEX form over EVEX when both exist) and freeze any residue as a small commented list. |
| `a64_table.zig` (3466 forms) | `AArch64.td` records + `InstAlias` | 2706 direct, 312 via aliases, 448 differ in field/immediate encoding | Map the 448 by operand class (immediate encoders: logical imm, `fpimm`, shifted/scaled offsets, SVE/SME index forms) so they come from LLVM; residue that LLVM encodes differently from Arm ARM is reviewed one by one and goes to an overrides file with the Arm ARM reference. |

Each table is a separate PR with: `check-tables` byte-exact, `compare_tables` old vs new (same form
set, or every difference listed), regenerated byte tests from clang, and tests for new or changed
forms. When all tables are LLVM-derived, a final PR removes AsmJit as a build-time dependency
(it stays only for the ported tests).

## Licensing

LLVM is Apache-2.0 WITH LLVM-exception. The generated table carries an attribution header and
`NOTICE` lists the derived files. AsmJit's license does not apply to `gp_table.zig` or the two
`isa.zig` files any more.
