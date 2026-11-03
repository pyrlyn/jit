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

The function is executed only when `--arch` equals the host architecture (and, for x86-64, the
host uses the SysV ABI, i.e. not Windows). Otherwise no executable memory is mapped: the code is
printed as hex on stdout, or written with `-o`.

Exit status: 0 success, 1 decode/compile/I/O error, 2 usage error.

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

## Lowering

Stack slot `i` is kept in a fixed caller-saved register; arguments stay in their ABI registers.

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

Each end-to-end case runs the CLI as a child process and checks its exit status and stdout, so
a wrong result or a crash fails the step. Foreign targets without an emulator are skipped.
