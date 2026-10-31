# Limitations

- Small instruction subset; 64-bit operands only on x86-64; no SSE/AVX/NEON.
- No Builder/Compiler, no register allocator, no formatter.
- Validation covers what the typed API cannot express (see table); it does not check CPU
  features or calling conventions.
- Hooks do not see raw bytes written with `CodeBuffer.emitBytes`, and `finalize` patches
  forward branches after `post` ran, so `ev.bytes` of a forward branch shows a zero displacement.
- One section; no relocations or base address (absolute calls go through `mov r11, imm; call r11`).
- `JitRuntime` maps a fresh 16 KiB region per function (no pooled allocator).

## ISA-database validation, CPU features, finalize and label checks

- `validate = true` additionally checks every emitted instruction against forms generated from AsmJit's ISA database (`scripts/gen_isa_tables.py` -> `src/x86_64/isa.zig`, `src/aarch64/isa.zig`); a mismatch rolls back and returns `error.EncodingMismatch`.
- `cpu_check = .target` with `target_features = &.{"BMI2", ...}` rejects instructions needing other features at compile time; `.host` checks against detected host features (CPUID / sysctl) at runtime (`error.UnsupportedCpuFeature`).
- Always on (all modes): `finalize` fails with `UnboundLabel` (list them with `unresolvedLabels`), `DisplacementOutOfRange` (rel8/rel32, AArch64 imm19/imm26) or `CodeTooLarge` (> 2 GiB) and leaves the code untouched; binding a label twice gives `LabelAlreadyBound`, an unknown label id gives `InvalidLabel`.
- Two AsmJit DB bugs are overridden in the generator (`cbz Xn`, `ret Xn` patterns).

## Alignment, traps, calling conventions, frames

- `alignNop(n)` pads with recommended multi-byte NOPs (x86-64) / `nop` words (AArch64); `alignTrap(n)` pads with `int3` / `udf #0`. `int3()` and `udf(imm16)` emit a single trap.
- `jit_runtime.add` fills the unused tail of the executable mapping with traps, so a stray jump past the code faults instead of running garbage.
- x86-64 `CallConv.sysv` / `.win64` / `.host`: argument, return and callee-saved registers, Win64 shadow space. `prolog(saved, locals)` / `epilog(saved, adj)` build an rbp frame keeping rsp 16-byte aligned.
- x86-64 `cc.stackArg(i)`: `[rbp + 16 + shadow + 8*(i - nregs)]` for stack-passed integer arguments (SysV and Win64).
- AArch64 `prolog(saved, locals)` / `epilog(saved, n)`: `stp x29, x30` frame, callee-saved registers (`CallConv.callee_saved`, x19..x28) pushed in `stp` pairs, up to 4080 bytes of locals. `CallConv.args` (x0..x7) and `CallConv.stackArgOffset(i)` (offset from x29, 8-byte slots; Apple's packing of sub-8-byte stack args is not modeled).

## Buffer limits

`CodeBuffer.max_size` (bytes, capped at 2 GiB) and `CodeBuffer.max_labels` can be lowered after `init`. Exceeding them returns `CodeTooLarge` (from growth, or at the latest from `finalize`) and `TooManyLabels` (from `newLabel`).

## Generated GP instruction set (x86-64)

`zig build gen-tables` (tools/gen_tables.zig) turns the legacy-encoded GP instructions of LLVM's X86 target description into `src/x86_64/gp_table.zig` (1067 forms, 201 mnemonics: ALU, shifts/rotates, mul/div, bit ops, movzx/movsx, cmov/set, xchg, bswap, popcnt/lzcnt/tzcnt, crc32, movbe, ...); see [llvm-tables.md](llvm-tables.md). Use it through `Assembler.inst`:

```zig
const gp = x.gp;
try a.inst(.movzx, .{ gp.r32(.rax), gp.byte(x.ptr(.rdi, 0)) });  // 8/16/32/64-bit registers and sized memory
try a.inst(.add, .{ gp.r16(.r9), @as(i64, 0x1234) });
try a.inst(.mov, .{ gp.r64(.rax), gp.ripLabel(table, .q) });     // RIP-relative, resolved at finalize
try a.inst(.lea, .{ gp.r64(.rcx), gp.rip(-7, .none) });
```

- Form selection: the shortest matching encoding, ties to MR, matching LLVM.
- `scripts/gen_gp_tests.py` writes `src/gp_tests.zig`: 3126 cases (every form with low/high/REX-only registers, SIB, RBP-like, RIP memory, immediates) checked byte-for-byte against LLVM.
- CPU features come from the database (`gp.requiredFeatures`). `Options.deny_features` rejects an instruction at compile time in every mode. `cpu_check = .target/.host` (with `validate`) checks the target or the host.
- Skipped: VEX/XOP/APX encodings, segment/control/debug registers, string instructions with implicit memory, moffs, far pointers and relative branches (use `jmp`/`jcc`/`call` with labels). `ah`..`bh` are not encodable. Hooks are called on the `inst` path as `Inst.gp` with `Ops.mn` (operands not decoded).
- An AsmJit DB bug is overridden: `shrd r/m, r, imm8` has a spurious 66 prefix.
- Differential fuzzing: `python3 scripts/fuzz_gp.py [cases=5000] [seed] [--target=x86_64-macos]` builds random forms (all registers, SIB/no-base/RIP memory, edge immediates), gets the bytes from LLVM, writes them to `src/gp_fuzz.zig`, runs `zig build test -Dgp-fuzz=true` and restores the placeholder. It found and fixed: `push 128..255` (6A sign-extends), `xchg eax, eax` (90 is `nop`, no zero-extension), and the 16-bit imm8 tie-break. Accepted, semantically identical deviations: `and r64, u32` drops REX.W (AsmJit), `xchg rax, rax` = `90`, `int 3` = `CD 03` (use `int3()` for `CC`).

## SIMD (SSE..AVX2, NEON basics)

x86-64, `x86_64/simd.zig`: `sse(&cb, ops.paddd, xmm(0), r(xmm(9)))` (legacy SSE..SSE4.1) and `vex(&cb, ops.paddd, ymm(0), ymm(1), m(ptr(.rdi, 0)))` (AVX/AVX2/FMA, `ymm` = VEX.L 256). Ops: moves (`movups/movaps/movdqu/movdqa`, `_st` = store), float add/sub/mul/div/min/max/sqrt/and/xor, scalar `addss/addsd/mulsd`, integer `padd*`, `psubd`, `pmullw`, `pmulld`, `pminsd`, `pand/por/pxor`, `pcmpeqd`, `pshufb`, `vfmadd231ps/pd`, `vpbroadcastd`, `vzeroupper`. Commutative VEX ops swap sources to use the 2-byte C5 prefix like LLVM.

AArch64, `aarch64/neon.zig`: `add/sub/mul` (16b/8h/4s/2d), `fadd/fsub/fmul/fmla` (4s/2d), `and/orr/eor/mov`, `ldr/str q`, `dup v.4s, w`, `umov w, v.s[i]`.

- Enable: import and call; nothing is compiled in unless used. `simd.feature(op, is_vex, wide)` names the CPU feature; check it against `cpu.host()` before running (the encoders do not check).
- Test: `src/simd_tests.zig` (generated by `python3 scripts/gen_simd_tests.py`): 817 x86 + 22 NEON cases, byte-exact vs LLVM. Execution tests in `src/tests.zig`: SSE2 always on x86-64, AVX2 only if the host has it (skipped under Rosetta 2, which has no AVX), NEON on arm64.
- Limitations: no EVEX/AVX-512, no shuffles/blends/converts/inserts yet, no SIMD through the `Assembler`/hooks/validation path (raw `CodeBuffer` emitters), no RIP-relative vector memory.

## Logger (text listing)

`x86_64.Logger(cap)` is a hook that writes an Intel-syntax listing with offsets and bytes:

```
0000  48 89 c8              mov rax, rcx
0003  48 83 c0 01           add rax, 1
0007  0f b6 07              movzx
```

- Enable: `AssemblerT(.{ .Hooks = x.Logger(4096) }).initHooks(&cb, &log)`, then `log.text()`. With the default `CallbackHooks`, forward events from your callback to a `Logger` instance.
- Disable: don't attach it (cost: one null check per instruction) or build with `-Dhooks=false` for the default assembler.
- Test: `x86_64 Logger: listing ...` in `src/tests.zig`.
- Limitations: fixed-size buffer (sets `truncated`); register names are 64-bit for the hand-written emitters; the generated `inst` path logs the mnemonic and bytes only; AArch64 keeps the generic `LogHooks` (id + bytes).

## Dual mapping (RW/RX views)

`jit_runtime.addDual(&cb)` (or `allocDual(size)`) maps one physical region twice: `f.rw` is read/write and never executable, `f.rx` is read/execute and never writable. Call `f.ptr(F)` to run, `f.write(offset, bytes)` to patch (icache flushed on AArch64), `f.release()` to unmap. No per-thread W^X toggling.

- macOS: anonymous RW mapping + `mach_vm_remap` (shared alias) + `mprotect` RX. Linux: `memfd_create` + two `MAP_SHARED` mappings. Other OSes: compile error; use `add` (single mapping, default, unchanged).
- Opt-in: `add` stays the default; nothing changes unless you call `addDual`.
- Test: `jit_runtime: dual mapping ...` in `src/tests.zig` runs code, patches through `rw`, checks the trap-filled tail.
- Limitations: no pooled allocator (one mapping per function, 16 KiB granule). Under Rosetta 2 patches through the alias are not observed (translation cache), so the test skips the patch check there (`underRosetta()`). Hardened-runtime apps without `com.apple.security.cs.allow-unsigned-executable-memory` may get `ProtectFailed`; use `add` (MAP_JIT) then. Linux is compile-checked only from the Mac.

## Sticky errors and error handler

Normal Zig style is `try a.addRR(...)`. For long generated sequences, `cb.keep(a.addRR(...))` records instead of returning. The first error is kept in `cb.sticky`, every error goes to `cb.error_handler` (`{ ctx, func(ctx, err, offset) }`), and `finalize` returns `EmitFailed` while a sticky error is set (`reset` clears it). The failing instruction emits nothing; later instructions are still emitted.
