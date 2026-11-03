# bc2jit

Compiles a KXBC bytecode function to machine code with pyrlyn-jit for one target selected at
run time, and runs it when that target is the host.

```
zig build bc2jit                          # -> zig-out/bin/bc2jit
zig-out/bin/bc2jit --arch aarch64 add.kxbc 40 2       # on an arm64 host: prints 42
zig-out/bin/bc2jit --arch riscv add.kxbc              # foreign target: prints the code as hex
zig-out/bin/bc2jit -a x86-64 -o add.bin add.kxbc      # raw machine code to a file
```

## Why KXBC

The KotlinX AoT pipeline (`newkotlinc`) has no bytecode of its own: it prints LLVM IR text
(`<output>.ll`, kept with `--emit-llvm`), runs LLVM `opt`, and links with clang. KXBC is the
proposed compact hand-off format the AoT side would also emit for code the JIT should run. It
lives here, next to the assemblers it targets; the KotlinX emitter is a separate change.

## Command line

`bc2jit -a ARCH [-o FILE] INPUT [ARG...]`, see `bc2jit --help`.

- `-a`, `--arch ARCH` (required; also `--arch=ARCH`): `x86-64`, `aarch64` or `riscv`, exact and
  case-sensitive. Missing, empty or unknown values print one line on stderr listing the allowed
  values and exit with status 2.
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
