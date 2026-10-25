# ZigJIT

A small AsmJit-style machine code generator in Zig 0.16 (stdlib only).

- Backends are separate modules, picked at build time: **x86-64** and **AArch64**
  (32-bit x86, RISC-V, MIPS: see [TODO-ARCHES.md](TODO-ARCHES.md)).
- Host and target are independent. You can emit x86-64 on an arm64 Mac; running the code
  needs target == host (x86-64 runs under Rosetta 2).
- No IR, no vtables, no runtime dispatch. Each method encodes one instruction directly into
  the buffer.
- Optional **safety layer**: operand validation and emission hooks. Both are chosen at
  comptime and cost nothing when off.

Design and comparison with AsmJit: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Roadmap (architectures, SIMD, Builder, Compiler, register allocator): [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start

```zig
const std = @import("std");
const zj = @import("zigjit");

var cb = zj.CodeBuffer.init(gpa);
defer cb.deinit();
const a = zj.x86_64.Assembler.init(&cb);

// fn add(a: i64, b: i64) i64   (SysV: rdi, rsi -> rax)
try a.movRR(.rax, .rdi);
try a.addRR(.rax, .rsi);
try a.ret();

try cb.finalize();                       // resolve forward-label fixups
const f = try zj.JitRuntime.add(&cb);    // copy into executable memory
defer f.release();
const add = f.ptr(*const fn (i64, i64) callconv(.c) i64);
std.debug.print("{d}\n", .{add(40, 2)});
```

## API

### Core
| Item | What it does |
|---|---|
| `CodeBuffer.init(gpa)` / `deinit()` | growable code buffer with labels and fixups |
| `cb.reset()` | drop code and labels, keep capacity (cheap reuse) |
| `cb.finalize()` | patch forward branches; returns `error.UnboundLabel` / `DisplacementOutOfRange` |
| `cb.code()` | the emitted bytes |
| `a.newLabel()`, `a.bind(label)` | labels; backward branches resolve at once, forward ones in `finalize` |
| `JitRuntime.add(&cb)` | map code executable (W^X; `MAP_JIT` on Apple arm64), `.ptr(T)`, `.release()` |

### x86-64 (`zj.x86_64`)
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

### AArch64 (`zj.aarch64`)
Registers `X` (`.x0`..`.x30`, `.zr_sp` = register 31, SP or XZR depending on the instruction).

| Method | Instruction |
|---|---|
| `movz`, `movk`, `movImm(rd, u64)` | 16-bit moves; `movImm` = movz + movk for non-zero halves |
| `mov`, `add`, `sub`, `mul`, `cmp` | register forms |
| `addImm`, `subImm`, `cmpImm` | 12-bit immediate forms (SP allowed) |
| `ldr`, `str` | `[xn, #off]`, `off` multiple of 8, < 32768 |
| `b`, `bl`, `bcond`, `cbz`, `cbnz`, `blr`, `ret`, `word(u32)` | branches; `word` emits a raw instruction |

## Safety

Validation and emission hooks. Both are optional, chosen at comptime, and free when off.

`Assembler` is `AssemblerT(default_options)`. Build your own variant with:

```zig
const A = zj.x86_64.AssemblerT(.{
    .validate = true,              // check operands before encoding
    .Hooks = zj.x86_64.CountingHooks, // or void
});
var counter: zj.x86_64.CountingHooks = .{};
const a = A.initHooks(&cb, &counter);   // A.init(&cb) = hooks type set but no instance
```

### Validation
With `.validate = true`, every emitter checks its operands and returns an error instead of
emitting wrong bytes. Nothing is written when a check fails.

| Backend | Checks | Error |
|---|---|---|
| x86-64 | `rsp` as index (unencodable: SIB index 100 means "none") | `InvalidOperand` |
| x86-64 | scale other than `.x1` without an index | `InvalidOperand` |
| both | label not created by this buffer (`jmp`, `call`, `jcc`, `b`, `bind`, ...) | `InvalidLabel` |
| AArch64 | register 31 (`.zr_sp`) where it would silently mean XZR: register-form `add/sub/mul/cmp`, `movz/movk`, `mov` destination, `ldr/str` data register, `blr`, `cbz/cbnz` | `InvalidOperand` |
| AArch64 | backward branch out of range (`b.cond`/`cbz`: ±1 MiB, `b`/`bl`: ±128 MiB) | `DisplacementOutOfRange` |
| AArch64 | misaligned `ldr/str` offset (checked always) | `InvalidImmediate` |

Immediate widths are enforced by the types (`i32` for imm32, `u12`, `u16`, `u15`, comptime
`shift`), so out-of-range literals fail to compile and runtime values must be cast.

```zig
const a = zj.x86_64.AssemblerT(.{ .validate = true }).init(&cb);
a.movRM(.rax, zj.x86_64.ptrIdx(.rdi, .rsp, .x1, 0)) catch |e| {
    // e == error.InvalidOperand
};
```

### Hooks
A hook type is a struct with optional methods. They are called directly (inlined), not
through function pointers.

```zig
pub fn pre(self: *H, ev: PreEvent) Error!void // before encoding; return an error to veto
pub fn post(self: *H, ev: Event) void         // after encoding
```

`Event` has `id: Inst` (e.g. `.mov_rr`, `.alu_ri`, `.add_imm`), `ops: Ops` (registers, memory,
immediate, label, ALU op / condition; unused fields are null), `offset: u32`, and
`bytes: []const u8` (the encoded instruction). `PreEvent` is the same without `bytes`.

Ready-made hooks (in both backends and in `zj.safety`):

| Hook | Purpose |
|---|---|
| `CountingHooks` | counts instructions and bytes |
| `LogHooks(cap)` | text log `0004  mov_rr   48 89 c8` into a fixed buffer (`.text()`) |
| `CallbackHooks` | runtime function pointer + context, e.g. for C callers (one indirect call per instruction) |

Custom hook example (logging + per-instruction counters), see `examples/example_hooks.zig`:

```zig
const Stats = struct {
    per_id: std.EnumArray(x.Inst, u32) = .initFill(0),
    pub fn post(self: *Stats, ev: x.Event) void {
        self.per_id.getPtr(ev.id).* += 1;
    }
};
var stats: Stats = .{};
const a = x.AssemblerT(.{ .Hooks = Stats }).initHooks(&cb, &stats);
```

Composite helpers report their parts: `callAbs` = `mov_ri` + `call_r`, `movImm` = `movz` + `movk`s.

### Enable / disable

| Build | `validate` default | `hooks` default |
|---|---|---|
| `Debug`, `ReleaseSafe` | on | on |
| `ReleaseFast`, `ReleaseSmall` | off | **on** |

Override with `-Dvalidate=true|false` and `-Dhooks=true|false`. They set
`default_options` of the default `Assembler` (`hooks` means "supports a `CallbackHooks`
instance"). `AssemblerT(.{...})` ignores the build flags, so code that needs a fixed
configuration can choose it explicitly.

Hooks stay compiled in for release builds so a hook can always be attached; `-Dhooks=false`
is the explicit opt-out. With hooks compiled in but no instance attached, each instruction
does one null check. That is free when the compiler can see that nothing is attached
(e.g. `init(&cb)` in the same function), and costs up to ~0.6–1 ns/instr in label- and
memory-heavy code when it cannot (see bench/RESULTS.md). Use `-Dhooks=false` or
`AssemblerT(.{})` for the fastest path.

When a layer is off it disappears at compile time: no checks, no branches, no extra field
(`@sizeOf(AssemblerT(.{})) == @sizeOf(*CodeBuffer)`), identical bytes.

## Building

Dependencies:
- **Zig 0.16.0**, pinned in `mise.toml` (`mise install`, then `mise exec -- zig ...` or an activated shell).
- No other dependencies for the library, tests and examples (stdlib only, links libc).
- Optional, only for the AsmJit benchmark and regenerating encoding tests: `clang` (Xcode
  command line tools), `objdump`, CMake + Ninja, Python 3, and an AsmJit checkout at `../asmjit-loc`.


```sh
zig build test                                     # all tests, native (arm64)
zig build test -Dtarget=x86_64-macos               # same tests under Rosetta
zig build test -Doptimize=ReleaseFast              # tests with optimizations
zig build run-example_aarch64                      # AArch64 JIT, executes natively
zig build run-example_x86_64 -Dtarget=x86_64-macos # x86-64 JIT, executes under Rosetta
zig build run-example_hooks                        # validation + hooks demo
zig build -Dtargets=aarch64                        # compile only the AArch64 backend
```

Tests:
- `src/enc_tests.zig` (generated by `scripts/gen_enc_tests.py`): ~2200 byte-exact cases, every
  instruction form across register classes (rsp/rbp/r12/r13/r8–r15, x29/x30/sp), all ALU ops,
  immediate boundaries (0, ±1, 127/−128/128/−129, i32 min/max, 0x80000000, 0xFFFFFFFF, imm64),
  and memory operands (every base with disp 0/disp8/disp32, every index × scale, no-base SIB).
  Expected bytes come from the LLVM assembler (`clang -c` + `objdump`). Regenerate after adding forms:
  `python3 scripts/gen_enc_tests.py`.
- `src/tests.zig`: validation (rejected operands, nothing emitted), hooks (counting, log text,
  veto, runtime callback, event contents), disabled layers give identical bytes and zero size,
  label/branch encodings (rel8/rel32, imm19/imm26, forward/backward).
- per-module tests in `src/*/assembler.zig` and `src/core/code_buffer.zig`.
- `bench/bench_x86_64.zig` tests: every benchmark scenario is byte-identical to AsmJit's output
  (`bench/ref/*.bin`) with layers off, validate on, and validate + counting hook.

## Benchmark vs AsmJit

```sh
# once: build AsmJit and the C++ harness
cmake -S ../asmjit-loc -B bench/asmjit-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DASMJIT_STATIC=ON && ninja -C bench/asmjit-build
c++ -O2 -std=c++20 -DASMJIT_STATIC -I ../asmjit-loc bench/asmjit_bench.cpp bench/asmjit-build/libasmjit.a -o bench/asmjit_bench

zig build -Doptimize=ReleaseFast -Dhooks=false && bench/run.sh 9 5000             # layers off
zig build -Doptimize=ReleaseFast -Dvalidate=true && bench/run.sh 9 5000          # validation
zig build -Doptimize=ReleaseFast && bench/run.sh 9 5000                          # default: hooks compiled in, none attached
ZJ_HOOK=1 bench/run.sh 9 5000                                                   # + runtime counting callback
zig build -Doptimize=ReleaseFast -Dbench-counting=true && bench/run.sh 9 5000    # comptime CountingHooks
```

Four scenarios (straight-line ALU/mov, calls, label loops, memory operands), ~1.3–1.6k
instructions each, buffer reset every iteration, x86-64 code generated on the arm64 host.
AsmJit runs with `kOptimizeForSize`, so both produce identical bytes (checked by `cmp`).
Results: [bench/RESULTS.md](bench/RESULTS.md).

## Limitations
- Small instruction subset; 64-bit operands only on x86-64; no SSE/AVX/NEON.
- No Builder/Compiler, no register allocator, no formatter.
- Validation covers what the typed API cannot express (see table); it does not check CPU
  features or calling conventions.
- Hooks do not see raw bytes written with `CodeBuffer.emitBytes`, and `finalize` patches
  forward branches after `post` ran, so `ev.bytes` of a forward branch shows a zero displacement.
- One section; no relocations or base address (absolute calls go through `mov r11, imm; call r11`).
- `JitRuntime` maps a fresh 16 KiB region per function (no pooled allocator).

### ISA-database validation, CPU features, finalize and label checks

- `validate = true` additionally checks every emitted instruction against forms generated from AsmJit's ISA database (`scripts/gen_isa_tables.py` -> `src/x86_64/isa.zig`, `src/aarch64/isa.zig`); a mismatch rolls back and returns `error.EncodingMismatch`.
- `cpu_check = .target` with `target_features = &.{"BMI2", ...}` rejects instructions needing other features at compile time; `.host` checks against detected host features (CPUID / sysctl) at runtime (`error.UnsupportedCpuFeature`).
- Always on (all modes): `finalize` fails with `UnboundLabel` (list them with `unresolvedLabels`), `DisplacementOutOfRange` (rel8/rel32, AArch64 imm19/imm26) or `CodeTooLarge` (> 2 GiB) and leaves the code untouched; binding a label twice gives `LabelAlreadyBound`, an unknown label id gives `InvalidLabel`.
- Two AsmJit DB bugs are overridden in the generator (`cbz Xn`, `ret Xn` patterns).

### Alignment, traps, calling conventions, frames

- `alignNop(n)` pads with recommended multi-byte NOPs (x86-64) / `nop` words (AArch64); `alignTrap(n)` pads with `int3` / `udf #0`. `int3()` and `udf(imm16)` emit a single trap.
- `jit_runtime.add` fills the unused tail of the executable mapping with traps, so a stray jump past the code faults instead of running garbage.
- x86-64 `CallConv.sysv` / `.win64` / `.host`: argument, return and callee-saved registers, Win64 shadow space. `prolog(saved, locals)` / `epilog(saved, adj)` build an rbp frame keeping rsp 16-byte aligned.
- x86-64 `cc.stackArg(i)`: `[rbp + 16 + shadow + 8*(i - nregs)]` for stack-passed integer arguments (SysV and Win64).
- AArch64 `prolog(saved, locals)` / `epilog(saved, n)`: `stp x29, x30` frame, callee-saved registers (`CallConv.callee_saved`, x19..x28) pushed in `stp` pairs, up to 4080 bytes of locals. `CallConv.args` (x0..x7) and `CallConv.stackArgOffset(i)` (offset from x29, 8-byte slots; Apple's packing of sub-8-byte stack args is not modeled).

### Buffer limits

`CodeBuffer.max_size` (bytes, capped at 2 GiB) and `CodeBuffer.max_labels` can be lowered after `init`. Exceeding them returns `CodeTooLarge` (from growth, or at the latest from `finalize`) and `TooManyLabels` (from `newLabel`).

### Generated GP instruction set (x86-64)

`scripts/gen_gp_tables.py` turns every legacy-encoded GP form in AsmJit's database into `src/x86_64/gp_table.zig` (1037 forms, 187 mnemonics: ALU, shifts/rotates, mul/div, bit ops, movzx/movsx, cmov/set, xchg, bswap, popcnt/lzcnt/tzcnt, crc32, movbe, ...). Use it through `Assembler.inst`:

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
- Skipped: VEX/XOP/APX encodings, segment/control/debug registers, string instructions with implicit memory, moffs, far pointers and relative branches (use `jmp`/`jcc`/`call` with labels). `ah`..`bh` are not encodable. Hooks are not called on the `inst` path.
- An AsmJit DB bug is overridden: `shrd r/m, r, imm8` has a spurious 66 prefix.
- Differential fuzzing: `python3 scripts/fuzz_gp.py [cases=5000] [seed] [--target=x86_64-macos]` builds random forms (all registers, SIB/no-base/RIP memory, edge immediates), gets the bytes from LLVM, writes them to `src/gp_fuzz.zig`, runs `zig build test -Dgp-fuzz=true` and restores the placeholder. It found and fixed: `push 128..255` (6A sign-extends), `xchg eax, eax` (90 is `nop`, no zero-extension), and the 16-bit imm8 tie-break. Accepted, semantically identical deviations: `and r64, u32` drops REX.W (AsmJit), `xchg rax, rax` = `90`, `int 3` = `CD 03` (use `int3()` for `CC`).

### Sticky errors and error handler

Normal Zig style is `try a.addRR(...)`. For long generated sequences, `cb.keep(a.addRR(...))` records instead of returning. The first error is kept in `cb.sticky`, every error goes to `cb.error_handler` (`{ ctx, func(ctx, err, offset) }`), and `finalize` returns `EmitFailed` while a sticky error is set (`reset` clears it). The failing instruction emits nothing; later instructions are still emitted.
