// AsmJit a64::Compiler side of bench/compiler (same sum/fact as bench_ir.zig).
// literal: one AsmJit instruction per IR op (constants in virtual registers, top-tested loop),
//          i.e. what a straightforward IR -> AsmJit Compiler translator emits;
// tuned:   hand-picked immediates and `add ..., lsl #3` (what a smarter translator would emit).
// compile = CodeHolder init + Compiler build + finalize (regalloc + serialize); full = + JitRuntime::add
// (AsmJit's pooled JitAllocator). Usage: asmjit_compiler_bench [dump_dir]
#include <asmjit/a64.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
using namespace asmjit;

static uint64_t now() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static uint64_t median(std::vector<uint64_t> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static void sum(a64::Compiler& cc, bool tuned) {
  FuncNode* fn = cc.add_func(FuncSignature::build<int64_t, const int64_t*, int64_t>());
  a64::Gp p = cc.new_gp64("p"), n = cc.new_gp64("n"), end = cc.new_gp64("end"), s = cc.new_gp64("s"), t = cc.new_gp64("t");
  fn->set_arg(0, p);
  fn->set_arg(1, n);
  Label loop = cc.new_label(), body = cc.new_label(), done = cc.new_label();
  if (tuned) {
    cc.add(end, p, n, a64::lsl(3));
    cc.mov(s, 0);
    cc.bind(loop);
    cc.cmp(p, end);
    cc.b_hs(done);
    cc.ldr(t, a64::ptr(p));
    cc.add(s, s, t);
    cc.add(p, p, 8);
    cc.b(loop);
  } else {
    a64::Gp c3 = cc.new_gp64("c3"), c8 = cc.new_gp64("c8");
    cc.mov(c3, 3);
    cc.lsl(end, n, c3);
    cc.add(end, p, end);
    cc.mov(s, 0);
    cc.mov(c8, 8);
    cc.b(loop);
    cc.bind(loop);
    cc.cmp(p, end);
    cc.b_lo(body);
    cc.b(done);
    cc.bind(body);
    cc.ldr(t, a64::ptr(p));
    cc.add(s, s, t);
    cc.add(p, p, c8);
    cc.b(loop);
  }
  cc.bind(done);
  cc.ret(s);
  cc.end_func();
}

static void fact(a64::Compiler& cc, bool tuned) {
  FuncNode* fn = cc.add_func(FuncSignature::build<int64_t, int64_t>());
  a64::Gp n = cc.new_gp64("n"), i = cc.new_gp64("i"), r = cc.new_gp64("r");
  fn->set_arg(0, n);
  Label loop = cc.new_label(), body = cc.new_label(), done = cc.new_label();
  cc.mov(i, 1);
  cc.mov(r, 1);
  if (tuned) {
    cc.bind(loop);
    cc.cmp(i, n);
    cc.b_hi(done);
    cc.mul(r, r, i);
    cc.add(i, i, 1);
    cc.b(loop);
  } else {
    a64::Gp one = cc.new_gp64("one");
    cc.mov(one, 1);
    cc.b(loop);
    cc.bind(loop);
    cc.cmp(i, n);
    cc.b_ls(body);
    cc.b(done);
    cc.bind(body);
    cc.mul(r, r, i);
    cc.add(i, i, one);
    cc.b(loop);
  }
  cc.bind(done);
  cc.ret(r);
  cc.end_func();
}

typedef int64_t (*SumFn)(const int64_t*, int64_t);
typedef int64_t (*FactFn)(int64_t);

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : nullptr;
  JitRuntime rt;
  const int64_t N = 10000000;
  std::vector<int64_t> data(N);
  for (int64_t i = 0; i < N; i++) data[i] = (int64_t)((uint64_t)i * 2654435761u) ^ (i >> 3);
  for (int tuned = 0; tuned < 2; tuned++) {
    const char* tag = tuned ? "tuned" : "literal";
    void* fns[2] = {};
    for (int k = 0; k < 2; k++) {
      std::vector<uint64_t> tc, tf;
      size_t bytes = 0;
      for (int it = 0; it < 1000; it++) {
        uint64_t t0 = now();
        CodeHolder code;
        code.init(rt.environment(), rt.cpu_features());
        a64::Compiler cc(&code);
        if (k == 0) sum(cc, tuned); else fact(cc, tuned);
        cc.finalize();
        tc.push_back(now() - t0);
        bytes = code.code_size();
      }
      for (int it = 0; it < 1000; it++) {
        uint64_t t0 = now();
        CodeHolder code;
        code.init(rt.environment(), rt.cpu_features());
        a64::Compiler cc(&code);
        if (k == 0) sum(cc, tuned); else fact(cc, tuned);
        cc.finalize();
        void* f = nullptr;
        rt.add(&f, &code);
        tf.push_back(now() - t0);
        rt.release(f);
      }
      CodeHolder code;
      code.init(rt.environment(), rt.cpu_features());
      a64::Compiler cc(&code);
      if (k == 0) sum(cc, tuned); else fact(cc, tuned);
      cc.finalize();
      if (dir) {
        char path[512];
        snprintf(path, sizeof path, "%s/asmjit_%s_%s.bin", dir, tag, k == 0 ? "sum" : "fact");
        FILE* fp = fopen(path, "wb");
        CodeBuffer& buf = code.text_section()->buffer();
        fwrite(buf.data(), 1, buf.size(), fp);
        fclose(fp);
      }
      rt.add(&fns[k], &code);
      printf("asmjit %s %s compile_ns=%llu full_ns=%llu bytes=%zu\n", tag, k == 0 ? "sum" : "fact",
             (unsigned long long)median(tc), (unsigned long long)median(tf), bytes);
    }
    std::vector<uint64_t> te;
    int64_t res = 0;
    for (int i = 0; i < 21; i++) { uint64_t t0 = now(); res = ((SumFn)fns[0])(data.data(), N); te.push_back(now() - t0); }
    printf("asmjit %s sum exec_ns=%llu result=%lld\n", tag, (unsigned long long)median(te), (long long)res);
    te.clear();
    int64_t acc = 0;
    for (int i = 0; i < 21; i++) {
      uint64_t t0 = now();
      uint64_t a = 0;
      for (volatile int64_t k = 0; k < 1000000; k++) a += (uint64_t)((FactFn)fns[1])(20 + (k & 1));
      te.push_back(now() - t0);
      acc = (int64_t)a;
    }
    printf("asmjit %s fact exec_ns=%llu result=%lld\n", tag, (unsigned long long)median(te), (long long)acc);
    rt.release(fns[0]);
    rt.release(fns[1]);
  }
  return 0;
}
