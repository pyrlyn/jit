#!/bin/sh
# IR compiler vs AsmJit a64::Compiler vs Cranelift on sum/fact (Apple Silicon), plus the C API
# fast path (opcode ids, pool). Everything is built under $OUT (default bench/compiler/out).
#   ASMJIT_SRC=<asmjit checkout> ASMJIT_LIB=<libasmjit.a built with -DASMJIT_STATIC=ON> \
#   [CRANELIFT=1 | CRANELIFT_BENCH=<cl-bench>] bench/compiler/run.sh [runs=3]
# CRANELIFT=1 builds bench/compiler/cranelift (cranelift-jit 0.137) with cargo. Without
# ASMJIT_SRC/ASMJIT_LIB the AsmJit side is skipped. Prints `uptime` before every run; compare
# only numbers from one session.
set -e
R=$(cd "$(dirname "$0")" && pwd)
RUNS=${1:-3}
OUT=${OUT:-$R/out}
mkdir -p "$OUT"
(cd "$R" && zig build -Doptimize=ReleaseFast)
(cd "$R/../.." && zig build lib -Doptimize=ReleaseFast --prefix "$OUT/pjlib")
clang -O2 -I "$R/../../include" "$R/capi_bench.c" "$OUT/pjlib/lib/libpyrlyn-jit.a" -o "$OUT/capi_bench"
if [ -n "$ASMJIT_SRC" ] && [ -n "$ASMJIT_LIB" ]; then
  c++ -O2 -std=c++20 -DASMJIT_STATIC -I "$ASMJIT_SRC" "$R/asmjit_compiler_bench.cpp" "$ASMJIT_LIB" -o "$OUT/asmjit_compiler_bench"
fi
if [ -z "$CRANELIFT_BENCH" ] && [ "${CRANELIFT:-0}" = 1 ]; then
  (cd "$R/cranelift" && CARGO_TARGET_DIR="$OUT/cl-target" cargo build --release -q)
  CRANELIFT_BENCH=$OUT/cl-target/release/cl-bench
fi
i=0
while [ "$i" -lt "$RUNS" ]; do
  echo "## run $i: $(uptime)"
  BENCH_DUMP="$OUT" "$R/zig-out/bin/bench-ir"
  if [ -x "$OUT/asmjit_compiler_bench" ]; then "$OUT/asmjit_compiler_bench" "$OUT"; fi
  "$OUT/capi_bench"
  if [ -n "$CRANELIFT_BENCH" ]; then "$CRANELIFT_BENCH" "$OUT"; fi
  i=$((i + 1))
done
