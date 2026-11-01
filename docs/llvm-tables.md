# Encoder tables from LLVM

`src/x86_64/gp_table.zig` (x86-64 legacy-encoded general-purpose instructions),
`src/x86_64/vec_table.zig` (every other x86-64 form: SSE, AVX, AVX-512, AVX10.2, MMX/3DNow!,
x87, AMX, mask, crypto and system instructions) and `src/aarch64/a64_table.zig` (AArch64 base,
NEON, SVE/SME) are generated from LLVM's target descriptions by a Zig tool, not from AsmJit.
LLVM is a generation-time input
only; the generated table is committed and a normal build needs neither LLVM nor the generator.

| Table | Source | Status |
|---|---|---|
| `src/x86_64/gp_table.zig` | LLVM `X86.td` (llvmorg-23.1.4) via `tools/gen_tables.zig` | generated from LLVM |
| `src/x86_64/vec_table.zig` | LLVM `X86.td` (llvmorg-23.1.4) via `tools/gen_tables.zig` | generated from LLVM |
| `src/aarch64/a64_table.zig` | LLVM `AArch64.td` (llvmorg-23.1.4) via `tools/gen_tables.zig` | generated from LLVM |
| `src/x86_64/isa.zig`, `src/aarch64/isa.zig` | AsmJit DB via `scripts/gen_isa_tables.py` | not yet ported |

## Regenerating

```sh
git clone --depth 1 --filter=blob:none --sparse -b llvmorg-23.1.4 https://github.com/llvm/llvm-project
git -C llvm-project sparse-checkout set llvm/include llvm/lib/Target/X86 llvm/lib/Target/AArch64
zig build gen-tables   -Dllvm-src=$PWD/llvm-project -Dllvm-tblgen=/path/to/llvm-tblgen   # writes the tables
zig build check-tables -Dllvm-src=$PWD/llvm-project -Dllvm-tblgen=/path/to/llvm-tblgen   # committed == fresh
python3 scripts/gen_gp_tests.py   # optional: regenerate src/gp_tests.zig (expected bytes from clang)
python3 scripts/gen_a64_table_tests.py --verify   # optional: src/a64_table_tests.zig (llvm-mc on PATH)
git show b5e4608:src/x86_64/vec_table.zig > /tmp/vec_table_asmjit.zig
python3 scripts/gen_vec_tests.py /tmp/vec_table_asmjit.zig /path/to/llvm-mc   # optional: src/testdata/vec_cases.txt
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
- `gen_x86_vec.zig`: every other `X86Inst` record (VEX/XOP/EVEX, and legacy records whose
  mnemonic is not in the GP table): operand kinds in Intel order (`VR128X` -> `xmm`, `VK*` -> `k`,
  `f128mem` -> `mem` 16, `vx64mem` -> VSIB, `RSTi` -> `st(i)`, ...), prefixes, map, opcode,
  VEX/EVEX `pp`/`W` (`IgnoresW` -> WIG)/`L`, ModRM roles (reg, rm, vvvv, is4, opcode register) from
  `Form`, the immediate size, EVEX `CD8_Scale` (disp8*N) and the mode. LLVM lists an EVEX
  instruction once per masking/broadcast/rounding variant (`rr`, `rrk`, `rrkz`, `rmb`, `rrb`, ...);
  these are merged into one form with the opmask (`{k}`/`{z}`), broadcast element size and
  `{er}`/`{sae}` flags. `{er}`/`{sae}` and disp8*N are copied between the register and memory
  forms of one encoding. x87 `st(i)` forms get the `C0+8*digit+i` opcode byte, x87 InstAliases
  (`fxch` = `fxch st(1)`, `faddp st(i)` = `faddp st(i), st`) and the `fwait` aliases (`fstsw` =
  `9B` + `fnstsw`) are added, `${cond}` records (`cmp${cond}xadd`) are expanded.
- `gen_a64.zig`: every AArch64 `Instruction` record with a 32-bit encoding (no pseudo /
  codegen-only) and every default-variant `InstAlias`. Fixed bits and operand field positions
  come from the record's `Inst` bit layout, operand kinds from the operand classes (the class ->
  kind/immediate-conversion list is the doc comment on `single`: `GPR64sp` -> `xsp`,
  `logical_imm64` -> `LogicalImm`, `simm9s16` -> scaled signed offset, `sve_fpimm_*` -> FP
  immediate, ...), the operand order from `AsmString`, tied operands from `Constraints`. Records
  that differ only in the element-size field (bits 22-23) are merged into one form with element
  lists. Alias operands that are constants in the alias dag become fixed field values.
- `gen_tables.zig`: entry point (`zig build gen-tables`, targets `x86` and `a64`).
- `compare_tables.zig`: form-by-form comparison of two tables (used for the migration below) and
  the `--exact` check behind `zig build check-tables`.

Project decisions that are not LLVM data live in `gen_x86_gp.zig` as small, commented lists:
which mnemonics this table offers (the rest is in `vec_table.zig`), the CPU feature per mnemonic
(LLVM's asm-level records rarely carry it), the reg,reg tie rule (MR before RM, RM for xchg),
`xchg` in both operand orders, the 16-bit-code-only variants that are left out, and a few manual
forms LLVM does not define (see below).

Project decisions for `vec_table.zig` live at the end of `gen_x86_vec.zig`, commented: the
mnemonic spellings kept from the previous table (the AVX10.2 BF16/FP8 names of the earlier ISE
revision such as `vaddnepbf16`, `fcomip`, `popa`, the string-op mnemonics), mnemonics not offered
(prefixes, Xeon Phi AVX512ER/PF/4FMAPS/4VNNIW, VIA PadLock, Key Locker wide forms, SGX leaf
functions, FRED, `wrpkru`, a few new extensions such as AVX512BMM), the GP-table mnemonics with
forms here (segment/control/debug `mov`/`push`/`pop`, `clzero`/`monitor`/`monitorx`/`pbndkb`),
the selection hints (`lowprio` for AVX10.2-only EVEX forms and GP `r/m` alternatives, `late_vex`
for AVX-VNNI/IFMA/NE-CONVERT), the form order for stack/far-branch forms (default operand size
first), a few SDM fixes (`rstorssp`/`clrssbsy` m64, `vmovw` m16), `cmpccxadd` kept in every mode
(the AsmJit x86 tests encode it in 32-bit mode) and manual forms (`vmgexit`, `pfrcpv`,
`pfrsqrtv`, `tmmultf32ps`, `invlpga` with the 67 prefix, `mov m16, sreg` with 66).

## Comparison with the previous (AsmJit-derived) tables

### gp_table.zig

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

### vec_table.zig

`zig run tools/compare_tables.zig -- OLD NEW` against the table at `b5e4608` (the comparison
reads `.w = 2` (WIG) as `.w = 0` and ignores `.n` on forms without a memory operand: both are
encoded identically):

```
OLD: 1537 mnemonics / 7131 forms, NEW: 1537 / 6895; identical 309, reordered (ties kept) 868,
tie order differs 18, differing 342, only-OLD 0, only-NEW 0
```

The tool's tie-order check is the GP encoder's rule (shortest encoding); `vec.zig` picks the first
matching form by preference level instead, so the order is checked by encoding instead:
`src/vec_table_tests.zig` runs 15747 cases (every form of both tables: plain, `{k}`, `{k}{z}`,
broadcast, `{er}`/`{sae}`, `{evex}`, and 32-bit mode for mode-independent legacy forms):

- 14842 `asmjit` cases: the bytes the old table produced, frozen; the new table gives the same.
- 147 `llvm` cases: the new table encodes differently or has a new form; the bytes equal llvm-mc
  23.1.4's.
- 758 `removed` cases: old forms LLVM does not define; they now fail with `NoMatchingForm`.

The AsmJit test suites (`-Dasmjit-tests`) and all other tests pass unchanged. What changed:

- Removed (not valid per the SDM/APM, not in LLVM): `{er}` on 128/256-bit forms (AVX10.2-256
  embedded rounding, dropped from the AVX10 spec) and `{sae}` likewise; `{z}` on memory-destination
  stores (`vmovups m, xmm {k}{z}`, `vpmov*`, `vcompress*`, `vextract*`, `vcvtps2ph`); masking on
  `vpinsr*` and `vmovw`; mixed-size BMI/TBM/`cmpccxadd`/RAO-INT forms (`andn r32, r64, ...`);
  `vpconflictq` with a 4-byte broadcast; one-operand `fcmov*`; SEAM instructions in 32-bit mode;
  `call r16`/`m16` in 32-bit mode; REX.W `mov sreg`/`sldt`/`smsw`/`str` memory forms and other
  operand-size variants AsmJit listed twice.
- Fixed encodings: `vcomx*`/`vucomx*` (`pp`), `vmpsadbw` EVEX (F3), `vgetexppbf16` (map 6, no
  prefix), `fsqrt` (D9 FA), x87 `st(i)` forms (`fadd`, `fcomi`, `fcmov*`, ...: `st(0), st(i)`
  operand order), `vgf2p8affine*qb`, `bextr` (TBM, XOP imm32), `vreduce*` with `{sae}`,
  `invlpgb`, `urdmsr`/`uwrmsr`.
- New forms: 16-bit `push`/`pop` of segment registers (66; picked only after the default form),
  `popa`/`pusha` with 66, `movss`/`movsd` reg-reg store forms, `maskmovdqu`, `vmovw r64`.
- Unsized memory where the old table had a size: `aes*kl`, `clflushopt`, `lds`/`les`/`lfs`/
  `lgs`/`lss`, `ldtilecfg`/`sttilecfg` (any size still matches; the size was not encoded).
- 3DNow! (including the Cyrix EMMI `pfrcpv`/`pfrsqrtv`, manual) stays in this table.

`scripts/gen_vec_tables.py` (the AsmJit generator) is no longer used for `vec_table.zig`.

## AArch64 (`a64_table.zig`)

Project decisions that are not LLVM data are commented lists at the end of `gen_a64.zig`:
aliases LLVM's assembler implements in C++ rather than as `InstAlias` (`mov Rd, #imm` via
MOVZ/MOVN/ORR, `lsl #n`, `bfi`/`bfxil`/`sbfiz`/`sbfx`/`ubfiz`/`ubfx`/`bfc`, `bti`, and the
`ic`/`dc`/`at`/`tlbi` `sys` aliases), `sxtb`/`sxth`/`sxtw Xd, Wn`, two AsmJit compatibility forms
(`sshll2`/`ushll2 Vd.2S, Vn.4S, #n`), form order hints (the encoder takes the first matching form),
the ADD/SUB `*Xrx64` records that the `*Xrx` forms cover, and the mnemonics kept (with no forms)
so existing code compiles.

`zig run tools/compare_tables.zig -- OLD NEW` against the AsmJit-derived table at `b5e4608`:

```
OLD: 1218 mnemonics / 3466 forms, NEW: 1497 / 5029; identical 347, reordered (ties kept) 34,
tie order differs 35, differing 802, only-OLD 0, only-NEW 279
```

Most "differing" forms are the same encodings in a different shape (LLVM field names, one form
per record instead of AsmJit's merged forms, element lists split or merged differently). The
behaviour is pinned by `src/a64_table_tests.zig` (generated from `src/testdata/a64_table_cases.txt`
by `scripts/gen_a64_table_tests.py`):

- `asmjit` (5123 cases): one case per old form (operands synthesized from the old table); the
  new table reproduces the old bytes.
- `llvm` (1923 cases): encodings checked with `llvm-mc -triple=aarch64 -mattr=+all`: the 10 old
  cases whose encoding changed, and new forms. The changed ones were AsmJit bugs:
  `fmov Zd.T, #imm` (6 cases, wrong imm8 encoding), `ldtrsb`/`ldtrsh Xt` (opc bit), and
  `ctermeq`/`ctermne Xn, Xm` (sz bit).
- `removed` (124 cases): operand combinations the old table accepted and llvm-mc rejects, which
  must be NoMatchingForm now: `st<op>a`/`st<op>al` (no such instructions), TME
  (`tstart`/`tcommit`/`tcancel`/`ttest`, removed from LLVM), the old `fcvtx`/`fcvtx2` spellings
  (`fcvtxn`/`fcvtxn2`), invalid element sizes (`andv`/`eorv Vd.D`, `clasta`/`clastb`, SVE
  narrowing shifts, `uqinc*`/`uqdec*` with a W destination) and invalid operands (`ldxp` with
  an offset, `ldaxrh`/`stlxrh Xt`, `stilp` post-index offsets, `gcspushx`/`gcspopx`/`gcspopcx`
  with a register, ...).
- 70 old mnemonics have no LLVM form (`kept_mnemonics`); they stay in `Mnemonic` and return
  NoMatchingForm. Besides the invalid ones above this includes spellings that need a keyword or
  register-list operand the encoder does not model: `brb`, `tlbip`, `gcsb`, `psb`, `tsb`,
  `cfp`/`cpp`/`dvp`/`cosp`, `mrrs`/`msrr`, `ld64b`/`st64b`/`st64bv`/`st64bv0`. `smstart`/`smstop`
  without an operand now encode as in LLVM (all of SVCR); the `sm`/`za` keyword forms are not
  offered.

The upstream AsmJit tests (`zig build test -Dasmjit-tests`) and the SVE/SME tests pass unmodified.

## Licensing

LLVM is Apache-2.0 WITH LLVM-exception. The generated tables carry an attribution header and
`NOTICE` lists the derived files. AsmJit's license does not apply to `gp_table.zig` and
`vec_table.zig` and `a64_table.zig` any more (the frozen `asmjit` expected bytes in
`src/testdata/vec_cases.txt` and `src/testdata/a64_table_cases.txt` are encodings, recorded from
the previous tables).
