#!/bin/sh
# IR compiler vs AsmJit Compiler (and optionally Cranelift): sum/fact on Apple Silicon.
#   ASMJIT_SRC=../asmjit-loc ASMJIT_LIB=.../libasmjit.a [CRANELIFT_BENCH=.../cl-bench] bench/compiler/run.sh [runs=3]
# Prints `uptime` before every process run; compare only numbers from one session.
set -e
R=$(cd "$(dirname "$0")" && pwd)
RUNS=${1:-3}
OUT=${OUT:-$R/out}
mkdir -p "$OUT"
(cd "$R" && zig build -Doptimize=ReleaseFast)
(cd "$R/../.." && zig build lib -Doptimize=ReleaseFast --prefix "$OUT/pjlib")
clang -O2 -I "$R/../../include" "$R/capi_bench.c" "$OUT/pjlib/lib/libpyrlyn-jit.a" -o "$OUT/capi_bench"
c++ -O2 -std=c++20 -DASMJIT_STATIC -I "$ASMJIT_SRC" "$R/asmjit_compiler_bench.cpp" "$ASMJIT_LIB" -o "$OUT/asmjit_compiler_bench"
i=0
while [ "$i" -lt "$RUNS" ]; do
  echo "## run $i: $(uptime)"
  BENCH_DUMP="$OUT" "$R/zig-out/bin/bench-ir"
  "$OUT/asmjit_compiler_bench" "$OUT"
  "$OUT/capi_bench"
  if [ -n "$CRANELIFT_BENCH" ]; then "$CRANELIFT_BENCH" "$OUT"; fi
  i=$((i + 1))
done
