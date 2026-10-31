# Benchmark vs AsmJit

```sh
# once: build AsmJit and the C++ harness
cmake -S ../asmjit-loc -B bench/asmjit-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DASMJIT_STATIC=ON && ninja -C bench/asmjit-build
c++ -O2 -std=c++20 -DASMJIT_STATIC -I ../asmjit-loc bench/asmjit_bench.cpp bench/asmjit-build/libasmjit.a -o bench/asmjit_bench

zig build -Doptimize=ReleaseFast -Dhooks=false && bench/run.sh 9 5000             # layers off
zig build -Doptimize=ReleaseFast -Dvalidate=true && bench/run.sh 9 5000          # validation
zig build -Doptimize=ReleaseFast && bench/run.sh 9 5000                          # default: hooks compiled in, none attached
PJIT_HOOK=1 bench/run.sh 9 5000                                                   # + runtime counting callback
zig build -Doptimize=ReleaseFast -Dbench-counting=true && bench/run.sh 9 5000    # comptime CountingHooks
```

Four scenarios (straight-line ALU/mov, calls, label loops, memory operands), ~1.3–1.6k
instructions each, buffer reset every iteration, x86-64 code generated on the arm64 host.
AsmJit runs with `kOptimizeForSize`, so both produce identical bytes (checked by `cmp`).
Results: [bench/RESULTS.md](../bench/RESULTS.md).

