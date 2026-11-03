#!/bin/bash
# Mutation run for pyrlyn/jit. Each mutant: one change to one line, written into a private copy of the tree
# (scripts/mutation/setup.sh), then `zig build test` (unit, encoder, execution, bc2jit unit + e2e, examples).
# usage: scripts/mutation/run.sh [file:max ...]   (default: the plan below)
# env: MUT_WORK (default ~/Grok/mutation-tests/runs), MUT_JOBS (2), MUT_TIMEOUT (600 s), ZIG_J (3)
set -eu
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
plan=${*:-"bc2jit/bytecode.zig:20 bc2jit/lower.zig:20 src/x86_64/assembler.zig:20 src/aarch64/assembler.zig:15 src/riscv/assembler.zig:15 src/core/code_buffer.zig:10 src/core/jit_runtime.zig:10 src/core/regalloc.zig:10"}
for fm in $plan; do
  f=${fm%%:*}; max=${fm##*:}
  work=${MUT_WORK:-$HOME/Grok/mutation-tests/runs}/jit-$(echo "$f" | tr / _)
  mkdir -p "$work"; export TMPDIR=${TMPDIR:-$work/tmp}
  nice -n 10 python3 "$here/mutate.py" run --root "$root" --file "$f" --work "$work" --max "$max" --jobs "${MUT_JOBS:-2}" \
    --timeout "${MUT_TIMEOUT:-600}" --setup "$here/setup.sh" --test "zig build test -j${ZIG_J:-3} --summary none 2>&1"
done
