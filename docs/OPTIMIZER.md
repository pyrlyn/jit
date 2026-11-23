# IR optimizer and tiered engine

`pyrlyn_jit.opt` rewrites an `ir.Func` in place into an equivalent, faster function that every
IR Compiler accepts. `pyrlyn_jit.engine` builds a tiered JIT on top of it: profiled tier-0 code,
then automatic tier-up to optimized code with runtime inlining.

## Levels

| Level | What runs |
| --- | --- |
| `.o0` | verify and type-check only |
| `.o1` | SSA construction, scalar rounds, constants to the entry block, out of SSA, block layout |
| `.o2` | `.o1` plus runtime inlining (with a resolver), LICM, the loop vectorizer (with a vector target) and loop rotation |

## Pipeline (`.o2`)

```
verify/typeCheck
-> inline                        (Options.resolver; hottest call sites first)
-> ssa.build                     (semi-pruned SSA with block parameters)
-> scalar rounds                 (sccp, instcombine, gvn, dce, simplifycfg; up to max_rounds)
-> licm
-> vectorize                     (Options.vector)
-> loops.rotate (+ licm again)
-> scalar rounds                 (when a loop pass changed something)
-> remat                         (i64 constants to block 0, deduplicated)
-> out_of_ssa                    (copy coalescing; no block parameters left)
-> layout                        (profile-guided or static block order)
```

Each pass has its module comment with the exact rules. In short:

| Pass | File | Does |
| --- | --- | --- |
| inline | `opt/inline.zig` | Replaces calls whose target has IR (`Resolver`) by the callee's body, within `InlineOptions` (size, hot count, caller budget, depth). |
| sccp | `opt/sccp.zig` | Wegman–Zadeck constant propagation, folds branches; evaluates with the interpreter's helpers. |
| instcombine | `opt/instcombine.zig` | Integer identities, reassociation, strength reduction (`mul` by 2^k, `sdiv`/`srem` by 2^k), copy propagation. |
| gvn | `opt/gvn.zig` | Dominator-scoped value numbering, redundant load elimination, store-to-load forwarding. |
| dce | `opt/dce.zig` | Mark-and-sweep dead code and dead block parameters. |
| simplifycfg | `opt/simplifycfg.zig` | Unreachable blocks, trivial parameters, jump threading, block merging. |
| licm | `opt/licm.zig` | Hoists invariant ops to preheaders; faulting ops only where they execute on every entry. |
| vectorize | `opt/vectorize.zig` | 128-bit vector loops (v2i64, v2f64, v4f32) with a scalar epilogue, integer reductions and runtime alias checks. |
| loops.rotate | `opt/loops.zig` | Top-tested loops become guarded bottom-tested loops (one branch per iteration). |
| remat | `opt/remat.zig` | i64 constants to the top of block 0, so the IR Compilers fold them into immediates. |
| out_of_ssa | `opt/out_of_ssa.zig` | Boissinot/Budimlić coalescing, hottest edges first; a counted loop updates its counter in place. |
| layout | `opt/layout.zig` | Pettis–Hansen chains: hot edges fall through, loops contiguous, cold blocks last. |

## Semantics

Equivalent means the same result, memory effects and calls in the same order for every input
on which the original is defined. The original is undefined where it executes `sdiv`/`srem` by
0 or `minInt / -1`, or a faulting load/store. The optimizer never speculates `sdiv`, `srem`,
`load`, `store` or `call` onto a path that did not execute them, but may delete an unused
`sdiv`/`srem`/`load`. Float folding happens only without NaN operands or results (NaN payloads
differ between hardware and the interpreter); float `min`/`max` are never commuted. The
reference interpreter `ir.interpret` is the oracle for every test.

## Zig API

```zig
const pjit = @import("pyrlyn_jit");
const stats = try pjit.opt.optimize(gpa, &f, .{
    .level = .o2,
    .vector = pjit.opt.VectorTarget.host, // .none disables the vectorizer
    .resolver = null,                     // ?Resolver: call target address -> callee IR
    .profile = null,                      // ?*const Profile: block and taken-edge counts
});
// stats: ops_before/after, blocks_after, inlined_calls, vectorized_loops, hoisted, rounds
```

`verify_each` (default on in Debug) re-checks structure, types and SSA after every pass.

## Engine

```zig
const e = try pjit.engine.Engine.create(gpa, .{ .hot_calls = 1000 });
defer e.destroy();
const id = try e.add(&f);                         // tier 0: instrumented code
const call = e.ptr(id, *const fn (i64) callconv(.c) i64);
_ = call(42);                                     // tiers up inside the 1000th call
```

- Every function has an immutable trampoline (`address`), so other JIT code can call it before
  and after tier-up; switching tiers is a release store into its slot.
- Tier 0 counts block entries and taken edges (one counter base register; about 1.8x slower
  than plain code on a tight loop). With `auto_tier_up`, the call that reaches `hot_calls`
  compiles tier 1; otherwise call `poll()` or `tierUp(id)`.
- Tier 1 is `opt.optimize` with the profile and a resolver that maps engine trampolines back to
  IR, so calls between engine functions are inlined. If it fails it is retried without
  vectors, then the function stays at tier 0 and `stats().last_error` records why.
- `reserve` + `define` allow mutual recursion; `profile(id)` returns the counts.

Limits: every compile maps at least one 16 KiB page and old tiers are freed only by `destroy`;
no deoptimization or re-profiling after tier-up; tier-0 counters are not atomic (counts may
be lost under concurrency, the hook still fires).

## Execution speed

`zig build bench-exec -Doptimize=ReleaseFast` on an Apple M3 Max (busy machine, minimum of 3
runs, ns per call). "Before" is the previous IR Compiler at `.o0`; "native" is the same
function compiled by Zig in ReleaseFast.

| Workload | before | `.o0` | `.o2` | `.o2` + inline | `.o2` + vectors | native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ir_sum_i64 | 2094 | 1564 | 1407 | 1371 | 1180 | 448 |
| ir_add_f64 | 2423 | 1463 | 1479 | 1540 | 839 | 944 |
| ir_saxpy_f32 | 2344 | 1274 | 1280 | 1288 | 359 | 747 |
| ir_dot_f64 | 5733 | 2904 | 2843 | 2852 | 2891 | 3185 |
| ir_calls | 1039 | 1161 | 1117 | 289 | 1120 | 1382 |
| ir_fib | 49431 | 36615 | 36170 | 36188 | 36398 | 18549 |
| kx_sum | 8517 | 6310 | 3627 | 3545 | 3565 | 1506 |
| kx_collatz | 54907 | 35171 | 11630 | 11664 | 11453 | 11972 |
| kx_gcd | 11029 | 9144 | 3791 | 3761 | 3730 | 2799 |

`dot_f64` stays scalar (float reductions would reassociate); `fib` recurses through a function
pointer the resolver cannot see. Inlining only applies with a resolver (`ir_calls`).

## Other entry points

- C API: `pjit_ir_compile_opt(buffer, src, len, arch, level)` (levels 0-2; vectors at level 2 on
  x86-64 and AArch64). See [c-api.md](c-api.md).
- bc2jit: `-O0`, `-O1`, `-O2` for KXBC v1 with the pyrlyn backend (`driver.Options.opt_level`).

## Testing

- Every pass has differential tests against `ir.interpret` (hand cases plus random programs
  from `opt/testgen.zig`).
- `src/opt_tests.zig` compares the interpreter, `.o1`, `.o2` and native code of the host IR
  Compiler on random programs and hand-written workloads. Longer runs:
  `PJIT_OPT_FUZZ=5000 zig build test-core -Dtest-filter=opt_tests`.
- `zig build bench-exec` measures generated code at `.o0`, `.o2`, `.o2` with inlining and
  `.o2` with vectors.
