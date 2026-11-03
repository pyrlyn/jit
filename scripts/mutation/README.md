# Mutation testing (pyrlyn/jit)

No source file in the repository is modified. `setup.sh` copies the tree (without `.git`, `zig-out`,
`.zig-cache`) into a private slot directory, `mutate.py` writes one mutated line into that copy, and
`zig build test` runs there: unit, encoder and table tests, execution tests, bc2jit unit tests and the
bc2jit e2e runs, examples.

```sh
scripts/mutation/run.sh                                     # the default plan (decoder, lowering, assemblers, core)
scripts/mutation/run.sh src/x86_64/assembler.zig:40 bc2jit/lower.zig:20   # file:mutants ...
MUT_JOBS=2 ZIG_J=3 MUT_TIMEOUT=600 MUT_WORK=~/Grok/mutation-tests/runs scripts/mutation/run.sh
python3 scripts/mutation/mutate.py list --root . --file bc2jit/lower.zig --max 20   # preview
python3 scripts/mutation/report.py ~/Grok/mutation-tests/runs/jit-* --survivors
```

- Operators: comparison flips, `and`/`or`, `+`/`-`, constants (0↔1, n±1, low bit of hex constants:
  opcode and encoding bits), `<<`/`>>`, `true`/`false`, negated `if`, deleted statements.
- Sampling: seeded shuffle (`--seed`, default 1), first N per file; reruns resume from `results.jsonl`.
- Outcomes: `killed` (failing tests recorded), `survived` (coverage gap or equivalent mutant),
  `timeout` (detected; the process group is killed), `invalid` (the mutant does not compile, e.g. an
  unused local after a deleted statement; excluded from the score).
- Needs the Zig from `mise.toml`. Each mutant rebuilds the test binaries (about 1-2 min).
