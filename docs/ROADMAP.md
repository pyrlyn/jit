# Roadmap (plan only, no code yet)

This is the single roadmap index. Detailed plans live in the linked sections.

| Topic | Detailed plan |
|---|---|
| x86 (32-bit) backend | [TODO-ARCHES.md §1](../TODO-ARCHES.md#1-x86-32-bit) |
| RISC-V backend | [TODO-ARCHES.md §2](../TODO-ARCHES.md#2-risc-v-rv64gc-first-rv32-later) |
| MIPS backend | [TODO-ARCHES.md §3](../TODO-ARCHES.md#3-mips-mips32mips64-r2-later-r6) |
| Other architectures (PowerPC64, LoongArch64, Wasm, AArch32) | [TODO-ARCHES.md §4](../TODO-ARCHES.md#4-others-only-if-needed) |
| x86-64 64-bit operands and immediates (8/16/32-bit sizes, RIP-relative, extension rules) | [TODO-ARCHES.md §5](../TODO-ARCHES.md#5-x86-64-64-bit-operands-and-immediates) |
| x86-64 SIMD (SSE..SSE4.2, AVX, AVX2/FMA, AVX-512, AVX10) | [TODO-ARCHES.md §6](../TODO-ARCHES.md#6-x86-64-simd) |
| x86 30-bit immediates | [TODO-ARCHES.md §7](../TODO-ARCHES.md#7-x86-30-bit-immediates) |
| Builder | [below, §1](#1-builder) |
| Compiler | [below, §2](#2-compiler) |
| Register allocator | [below, §3](#3-register-allocator) |
| Effort estimates vs AsmJit parity | [ARCHITECTURE.md](ARCHITECTURE.md#remaining-to-reach-asmjit-parity-realistic-estimate-one-experienced-engineer) |

---

Ordering: Builder -> Compiler (no RA, fixed registers) -> linear-scan RA -> optional graph
coloring. Each layer is generic over the target with comptime parameters, like `AssemblerT`.

## 1. Builder

Goal: emit instructions into a node list instead of bytes, run passes over it, then serialize
to an Assembler. Users never write bytes or compute offsets.

### Design
- `BuilderT(comptime Arch: type, comptime opts: Options)`. `Arch` is the backend module
  (`zigjit.x86_64`), so `Arch.Inst`, `Arch.Ops`, `Arch.AssemblerT` are known at comptime.
- Node: `{ kind: enum { inst, label, align, embed, comment, func, sentinel }, inst: Arch.Inst,
  ops: Arch.Ops, next, prev }`, stored in an arena with indices (u32) instead of pointers.
- API mirrors the Assembler (`b.movRR(...)`) but records nodes. It reuses the same `Inst`/`Ops`
  types as the safety layer, so validation and hooks work unchanged.
- Passes: `fn run(b: *Builder) !void` functions passed as a comptime tuple
  (`b.runPasses(.{ removeDeadCode, shortenBranches })`); no pass manager with vtables.
- Serialization: walk nodes, `switch (node.inst)` -> call the matching Assembler method
  (generated `inline else` switch over `Inst`, one direct call per node).
- Labels are nodes, so branch shortening (rel8 for forward targets) becomes possible.

### Steps and milestones
1. Node storage, labels, insertion cursor (before/after node). 3 days.
2. Record + serialize every current instruction; output identical to direct Assembler. 3 days.
3. Passes: dead code after unconditional jump, forward branch shortening, align/embed. 4 days.
4. Formatter: print node list (reuses `LogHooks` naming). 2 days.
Estimate: ~2 weeks.

### Tests
- for every scenario/example: Builder output == Assembler output (byte-exact);
- node edits (insert/remove before serialize) produce expected bytes;
- branch shortening: forward rel8 where possible, matches AsmJit Builder output;
- fuzz: random instruction streams through Builder and Assembler, compare bytes.

## 2. Compiler

Goal: compile a minimal IR to machine code through the Builder.

### IR
- Typed values `i32`, `i64`, `ptr` (later f32/f64, vectors).
- Functions of basic blocks; three-address instructions in SSA-ish form (block parameters
  instead of phi nodes):
  `arith (add sub mul and or xor shl shr sar)`, `cmp (eq ne lt le gt ge, signed/unsigned)`,
  `load`, `store`, `const`, `call`, `ret`, `br`, `cbr`.
- Built with an `IrBuilder` API; verifier checks types, dominance, terminators.

### Lowering
- Instruction selection by pattern matching per IR op (`comptime` tables per arch):
  `add v, const` -> `aluRI`, `load [p + c]` folded into memory operands, `cmp+cbr` -> `cmp; jcc`.
- Virtual registers in Builder nodes (`Ops` extended with `VReg` ids), replaced by the RA.
- Calling conventions as data: SysV x86-64 and Win64 first, AAPCS64 later:
  argument registers, return registers, callee-saved set, stack alignment, shadow space (Win64), red zone.
- Frame: `FuncFrame` computes stack size, saved registers, alignment; prolog/epilog inserted
  as Builder nodes after RA (so only used callee-saved registers are saved).

### Steps and milestones
1. IR data structures, builder, verifier, printer. 1 week.
2. Lowering with a trivial allocator (every value in a stack slot). Runs real functions. 1 week.
3. Calling conventions + frames (SysV, Win64), calls to absolute addresses and labels. 1 week.
4. Instruction-selection improvements (immediates, memory operand folding, cmp/branch fusion). 1 week.
5. AArch64 lowering (AAPCS64). 1–2 weeks.
Estimate: 3–4 weeks for x86-64, +1–2 weeks AArch64.

### Tests
- unit: verifier rejects bad IR; printer round-trip;
- byte-exact for small functions with fixed register choices;
- execution under Rosetta/native: arithmetic, loops, recursion (fact, fib), calls into Zig,
  many arguments (stack-passed), mixed i32/i64;
- differential: same IR compiled by ZigJIT and an interpreter (or AsmJit Compiler via a
  translator), compare results on random inputs.

## 3. Register allocator

**Status (feat/regalloc): milestone 1 + call clobbers + fixed registers** in
`src/core/regalloc.zig` (`zigjit.regalloc`): arch-agnostic (register masks), bitset liveness
over the CFG, one interval per vreg, linear scan with furthest-end spilling and spill-slot
reuse, callee-saved-only for call-crossing values, pre-colored vregs, and an interference
checker (`verify`) exercised by a random-program fuzz test. Not yet: splitting, lifetime
holes, edge move resolution, loop-depth spill weights, two-address hints, graph coloring.

### Choice
- Default: **linear scan** (Poletto/Sarkar with lifetime holes and splitting, as in V8/HotSpot
  C1): fast, good enough for a JIT.
- Optional: **graph coloring** (Chaitin-Briggs with coalescing) behind `Options.ra = .graph`,
  for slower but better code. AsmJit uses a local+global allocator in between.

### Pieces
1. Liveness: CFG, block order (reverse post-order, loops kept contiguous), live-in/out sets
   with bitsets, live ranges with holes per virtual register.
2. Virtual vs physical registers: `VReg{ id, class: gp|vec, size }`; physical register sets per
   arch as comptime bitsets.
3. Fixed constraints: x86 `div/idiv` (rax, rdx), shifts by `cl`, call arguments/return values,
   `imul` one-operand forms; AArch64 has few. Modeled as pre-colored ranges at specific positions.
4. Calls: caller-saved registers are clobbered at call sites (live ranges crossing a call
   prefer callee-saved registers or are spilled around the call).
5. Spilling and reload: spill weight = uses × loop depth; spill slots allocated in `FuncFrame`;
   split ranges at block boundaries; resolve moves on CFG edges (parallel-move resolution with
   cycle breaking via xchg or a temp).
6. Two-address fix-up for x86 (`dst = src1 op src2` -> `mov dst, src1; op dst, src2`) with hints
   to avoid the extra move.
7. Integration: Compiler emits Builder nodes with vregs -> RA pass rewrites operands and inserts
   spill/reload/move nodes -> prolog/epilog -> serialize to Assembler.
8. AArch64: 31 GP registers, almost no fixed constraints, `x18` reserved on Apple, `x29/x30`
   frame/link registers; same allocator with a different register file.

### Steps and milestones
1. Liveness + linear scan without splitting (spill whole range). 2 weeks.
2. Fixed-register constraints and call clobbers. 1–2 weeks.
3. Splitting, edge move resolution, spill-weight heuristics. 2–3 weeks.
4. Two-address hints, coalescing of moves. 1 week.
5. Optional graph coloring. 2–3 weeks.
Estimate: 6–10 weeks total.

### Tests
- unit: liveness on hand-built CFGs; interference correctness checker (no two simultaneously
  live vregs share a register) run after every allocation in Debug;
- spill stress: functions with 50+ live values, register file artificially limited to 3–4
  registers via `Options` to force spills;
- fixed constraints: div/shift/call sequences produce correct values;
- execution: compiled IR vs interpreter on random programs (fuzzing, thousands of cases);
- differential vs AsmJit Compiler on the same functions: results equal, code size and spill
  counts reported (not required to match);
- compile-time benchmarks: allocation time per instruction vs AsmJit Compiler.
