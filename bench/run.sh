#!/bin/sh
# Runs both benchmarks RUNS times; per scenario prints median(avg ns/instr) and min(min ns/instr).
# usage: bench/run.sh [runs] [iters]   (after `zig build -Doptimize=ReleaseFast` and building asmjit_bench)
set -e
cd "$(dirname "$0")"
RUNS=${1:-7}; ITERS=${2:-5000}
mkdir -p out; : > out/raw.txt
for i in $(seq "$RUNS"); do
  ASMJIT_OPT_SIZE=1 ./asmjit_bench "$ITERS" out >> out/raw.txt
  ../zig-out/bin/bench_x86_64 "$ITERS" out >> out/raw.txt 2>&1
done
field() { sed -n "s/.* $1=\([^ ]*\).*/\1/p"; }
for s in straight calls loops mem; do
  cmp -s out/asmjit_$s.bin out/zigjit_$s.bin && same=identical || same=DIFFERENT
  for lib in asmjit zigjit; do
    L=$(grep "^$lib $s " out/raw.txt)
    med=$(echo "$L" | field avg_ns_per_instr | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
    mn=$(echo "$L" | field min_ns_per_instr | sort -n | head -1)
    b=$(echo "$L" | head -1 | field bytes)
    printf "%-8s %-6s bytes=%-6s median=%6s ns/instr  min=%6s ns/instr  output:%s\n" $s $lib $b $med $mn $same
  done
done
