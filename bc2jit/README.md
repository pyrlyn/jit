# bc2jit

Compiles a KXBC bytecode function to machine code with pyrlyn-jit for one target selected at
run time, and runs it when that target is the host.

```
zig build bc2jit                          # -> zig-out/bin/bc2jit
zig-out/bin/bc2jit --arch aarch64 add.kxbc 40 2       # on an arm64 host: prints 42
zig-out/bin/bc2jit --arch riscv add.kxbc              # foreign target: prints the code as hex
zig-out/bin/bc2jit -a x86-64 -o add.bin add.kxbc      # raw machine code to a file
```

## Zig and C APIs

The `bc2jit` module and C API are enabled by default. Disable bytecode support with
`zig build -Dbc2jit=false` (`PYRLYN_JIT_BC2JIT=OFF` in CMake). The core assembler/JIT remains
available; the CLI steps are omitted, the Zig `compile` returns `error.FeatureDisabled`, and
C symbols stay linkable as stubs returning `PJIT_BC2JIT_DISABLED`.

In a Zig build, import `dependency.module("bc2jit")` as `bc2jit`:

```zig
const bc2jit = @import("bc2jit");
var diag: bc2jit.Diagnostic = .{};
var compiled = try bc2jit.compile(gpa, .aarch64, kxbc_bytes, .{}, &diag);
defer compiled.deinit();
if (compiled.entry()) |f| {
    const answer = f(40, 2, 0, 0);
    _ = answer;
}
```

`compile` validates the complete KXBC v0/v1 module and owns all resulting allocations.
Input bytes may be freed after it returns. `code()` borrows the finalized machine code;
`entry()` returns a C-ABI function pointer only for a compatible host target. Pass
`.{ .executable = false }` to emit code without mapping executable memory. Release the result
with `deinit()` after all calls have finished. `diag.offset` identifies decoder errors;
`errorMessage(err)` describes decoder and compiler failures.

C consumers include `<bc2jit.h>` and link the existing `libpyrlyn-jit`:

```c
pjit_bc2jit *compiled = NULL;
pjit_bc2jit_diagnostic diagnostic;
int status = pjit_bc2jit_compile(bytes, length, PJIT_BC2JIT_HOST, 0,
                                &compiled, &diagnostic);
if (status == PJIT_BC2JIT_OK) {
    pjit_bc2jit_fn entry = pjit_bc2jit_entry(compiled);
    if (entry) printf("%lld\n", (long long)entry(40, 2, 0, 0));
    pjit_bc2jit_free(compiled);
}
```

The `PJIT_BC2JIT_NO_JIT` flag emits code only; `pjit_bc2jit_code` exposes its bytes.
The handle owns both code and the optional executable mapping. The entry and code pointers
remain valid until `pjit_bc2jit_free`; input bytes are borrowed only during compilation.
`pjit_bc2jit_enabled()` reports whether the library includes bytecode support.

Measure API paths and enabled/disabled library/binary sizes with
`python3 scripts/bench_bc2jit_api.py`; see [the recorded results](../bench/BC2JIT_API_RESULTS.md).

## Why KXBC

The KotlinX AoT pipeline (`newkotlinc`) has no bytecode of its own: it prints LLVM IR text
(`<output>.ll`, kept with `--emit-llvm`), runs LLVM `opt`, and links with clang. KXBC is the
proposed compact hand-off format the AoT side would also emit for code the JIT should run. It
lives here, next to the assemblers it targets; the KotlinX emitter is a separate change.

## Command line

`bc2jit -a ARCH [--no-jit] [-o FILE] INPUT [ARG...]`, see `bc2jit --help`.

- `-a`, `--arch ARCH` (required; also `--arch=ARCH`): `x86-64`, `aarch64` or `riscv`, exact and
  case-sensitive. Missing, empty or unknown values print one line on stderr listing the allowed
  values and exit with status 2.
- `--no-jit`: emit code without mapping or executing it, including for the host target.
- `-o`, `--output FILE`: write the raw machine code to FILE.
- `ARG...`: up to four integers (decimal, or `0x`/`0o`/`0b` prefixed, may be negative) passed to
  the function when it is executed; missing ones are 0.
- `--` ends option parsing; `-h`/`--help` prints the usage.

The arch is chosen per run. All backends are compiled in by default and dispatched on the flag;
x86-64 and AArch64 can be left out with `-Dtargets=` (RISC-V is always built), in which case
selecting them fails with exit status 1 and a message naming the build option.

The function is executed only when `--arch` equals the host architecture (and, for a v0 module on
x86-64, the host uses the SysV ABI, i.e. not Windows; v1 modules use the host C ABI, Win64 included). Otherwise no executable memory is mapped: the code is
printed as hex on stdout, or written with `-o`.

Exit status: 0 success, 1 decode/compile/I/O error, 2 usage error.

## LLVM backend

The optional backend uses the LLVM **C API**, in process: no `opt`, `clang` or `lli`
subprocesses. Both KXBC versions use the existing validated lowering to integer IR, then LLVM
IR construction and `LLVMRunPasses`. The optimized module runs in either LLVM MCJIT or LLVM
Interpreter. The original `pyrlyn` backend remains the default.

Build against an LLVM installation with headers and a shared `libLLVM` that includes MCJIT,
Interpreter and PassBuilder (tested with LLVM 23.1.5):

```sh
zig build bc2jit -Dllvm-prefix=/opt/homebrew/opt/llvm  # Homebrew example
zig-out/bin/bc2jit -a aarch64 --backend llvm-jit add.kxbc 40 2
zig-out/bin/bc2jit -a aarch64 --backend llvm-interpreter add.kxbc 40 2
zig-out/bin/bc2jit -a aarch64 --backend llvm-interpreter --no-opt add.kxbc 40 2
zig build test-bc2jit e2e-bc2jit -Dllvm-prefix=/opt/homebrew/opt/llvm
```

Use the host's `--arch` value (`x86-64`, `aarch64`, or `riscv`). These LLVM modes execute on
the host; foreign `--arch` and raw machine-code output (`-o`) are rejected as usage errors.
Without the build option, selecting LLVM reports how to enable it and exits with status 1.

Optimization is enabled by default: `function(mem2reg,sccp,adce,simplifycfg)` promotes stack
slots/locals to SSA, folds constants, removes dead computations and simplifies control flow.
This deliberately scalar pipeline is shared by both execution modes; it is not LLVM's full
`default<O2>` pipeline. MCJIT also uses code-generation optimization level 2 (0 with `--no-opt`).
The backend preserves wrapping arithmetic, shifts modulo 64, zero-initialized locals and
KXBC's total division/remainder rules. Modules are verified before and after optimization.
See the [LLVM pass API](https://llvm.org/doxygen/group__LLVMCCoreNewPM.html).

The Zig API optimizes once and reuses the resulting program for subsequent calls:

```zig
// f is a Function returned by bc2jit.bytecode.decode.
var diag: pjit.llvm.Diagnostic = .{};
var program = try bc2jit.driver.compileLlvm(gpa, f, .{
    .mode = .interpreter, // or .jit
    .optimize = true,
}, &diag);
defer program.deinit();
const result = program.call(.{ 40, 2, 0, 0 });
// On failure, diag.message() contains a bounded copy of LLVM's diagnostic, if any.
// program.irText(gpa) returns an owned LLVM IR snapshot; free it with gpa.free.
```

`pjit.llvm.prepare` also accepts the integer mutable-register IR subset emitted by KXBC
lowering (`const`, `copy`, integer arithmetic, `br`, `cbr`, `ret`, at most four parameters).
Other IR operations are rejected. Programs own their LLVM context, module and engine;
input IR/bytecode can be released after preparation. Serialize calls on the same interpreter
program. LLVM Interpreter has no instruction-budget API: unlike `bytecode.eval(max_steps)`,
it can run indefinitely on a nonterminating input, just like JIT execution. The existing
bounded reference interpreter remains available.

LLVM-enabled tests compare both modes, with optimization on and off, against `bytecode.eval`
for arithmetic edge cases, all comparisons, loops, a 64-deep stack and 400 random programs.
They also verify actual constant folding in the emitted LLVM IR, allocation-failure cleanup,
repeated calls, invalid IR rejection and CLI execution for both KXBC versions.

## Status

Not a draft any more: the format is versioned (v0 frozen, v1 current), the decoder validates
every module completely, both lowerings are tested against a reference interpreter
(`bytecode.eval`) on every target, and the CLI contract (flags, exit codes, output) is covered by
end-to-end tests. Not included: the KotlinX side that emits KXBC (separate change), calls,
floating point, more than four arguments.

## Format (KXBC v0)

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `KXBC` |
| 4 | 1 | version, `0` |
| 5 | 1 | argument count, 0..4 |
| 6 | 2 | code length in bytes, little-endian |
| 8 | n | code |

One function over `i64` with an operand stack (depth at most 5):

| opcode | name | operand | effect |
|---|---|---|---|
| `0x01` | `push_i64` | i64 LE | push immediate |
| `0x02` | `push_arg` | u8 | push argument n |
| `0x10` | `add` | | pop b, a; push a + b (wrapping) |
| `0x11` | `sub` | | pop b, a; push a - b |
| `0x12` | `mul` | | pop b, a; push a * b (low 64 bits) |
| `0x20` | `dup` | | push copy of top |
| `0x21` | `drop` | | pop |
| `0xF0` | `ret` | | return top; must be last |

The decoder validates the whole module (opcodes, operands, argument indices, stack depth, final
`ret`) and reports the byte offset of the first error.

## Format (KXBC v1)

Same header with version `1`. All v0 opcodes plus:

| opcode | name | operand | effect |
|---|---|---|---|
| `0x13` | `div` | | pop b, a; push a / b truncated; `a / 0 = 0`, `minInt / -1 = minInt` |
| `0x14` | `rem` | | pop b, a; push a % b (sign of a); `a % 0 = a`, `a % -1 = 0` |
| `0x15` `0x16` `0x17` | `and` `or` `xor` | | bitwise |
| `0x18` `0x19` `0x1A` | `shl` `shr` `sar` | | shift a by b mod 64 (`shr` logical, `sar` arithmetic) |
| `0x22` | `swap` | | exchange the two top values |
| `0x30` | `load_local` | u8 n < 16 | push local n (locals start at 0) |
| `0x31` | `store_local` | u8 n < 16 | pop into local n |
| `0x40` | `jmp` | u16 LE code offset | jump |
| `0x41` | `br_if` | u8 cond, u16 LE offset | pop b, a; jump if cond(a, b) |

Conditions: `0` eq, `1` ne, `2` lt, `3` le, `4` gt, `5` ge (signed), `6` ult, `7` ule, `8` ugt,
`9` uge. Every operation is total (no traps); arithmetic wraps.

Validation (data flow over the control-flow graph, error offset = offending instruction):
jump targets must be instruction starts (`BadJumpTarget`); every instruction must be reachable
(`UnreachableCode`); all paths into an instruction must have the same stack depth
(`StackMismatch`); depth at most 64; control must not fall off the end (`MissingRet`); `ret`
may appear anywhere.

## Lowering

v1 modules are lowered through the pyrlyn-jit IR (`lower_ir.zig`): every stack slot, local and
argument becomes a mutable IR vreg, basic blocks start at jump targets and after branches, and
the x86-64, AArch64 or RV64 Compiler does register allocation (linear scan with spilling, so
all 64 stack slots work), the host C ABI and branch lowering. `div`/`rem` get guard blocks for
`b == 0` and `b == -1`, so results are identical on every target. The same path also compiles
v0 functions (tested to give the same results as the direct lowering below).

v0 modules keep the direct lowering: stack slot `i` is kept in a fixed caller-saved register; arguments stay in their ABI registers.

| | args | slots 0..4 | return |
|---|---|---|---|
| x86-64 (SysV) | rdi rsi rdx rcx | rax r10 r11 r8 r9 | rax |
| AArch64 (AAPCS64) | x0..x3 | x9..x13 | x0 |
| RV64 (LP64) | a0..a3 | t0..t4 | a0 |

With `d` the stack depth before the instruction and `S` the slot registers:

| opcode | x86-64 | AArch64 | RV64 |
|---|---|---|---|
| `push_i64 v` | `movRI S[d], v` | `movImm S[d], v` (movz/movk) | `li S[d], v` |
| `push_arg n` | `movRR S[d], arg[n]` | `mov S[d], arg[n]` | `mv S[d], arg[n]` |
| `add` | `aluRR add S[d-2], S[d-1]` | `add S[d-2], S[d-2], S[d-1]` | `add S[d-2], S[d-2], S[d-1]` |
| `sub` | `aluRR sub S[d-2], S[d-1]` | `sub S[d-2], S[d-2], S[d-1]` | `sub S[d-2], S[d-2], S[d-1]` |
| `mul` | `imulRR S[d-2], S[d-1]` | `mul S[d-2], S[d-2], S[d-1]` | `mul S[d-2], S[d-2], S[d-1]` |
| `dup` | `movRR S[d], S[d-1]` | `mov S[d], S[d-1]` | `mv S[d], S[d-1]` |
| `drop` | (nothing) | (nothing) | (nothing) |
| `ret` | `movRR rax, S[d-1]`; `ret` | `mov x0, S[d-1]`; `ret` | `mv a0, S[d-1]`; `ret` |

Same-register moves are omitted.

## Tests

```
zig build test-bc2jit                                     # unit tests (part of zig build test)
zig build e2e-bc2jit                                      # CLI on fixtures, --arch = host
zig build e2e-bc2jit -Dtarget=x86_64-macos -frosetta      # x86-64 on Apple Silicon
zig build e2e-bc2jit -Dtarget=riscv64-linux-musl -fqemu   # RV64 under qemu-user (CI)
```

Unit tests include `tests_v1.zig`: v1 decode/encode round trip and depths, every validation
error with its offset, interpreter semantics, every binary op and condition on edge values
(minInt, -1, 0, shift counts 63..65) executed on the host vs `eval`, factorial/gcd loops, a
64-deep stack, all targets lowering, v0 parity between the two lowerings, and a 400-program
random differential fuzz (forward branches, a counted loop, locals). The mutation fuzz in
`tests_extra.zig` also lowers every accepted v1 module on every target.

Each end-to-end case runs the CLI as a child process and checks its exit status and stdout, so
a wrong result or a crash fails the step. Foreign targets without an emulator are skipped.
