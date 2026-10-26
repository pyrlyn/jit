// AsmJit side of the ZigJIT benchmark. Same instruction streams as bench_x86_64.zig.
#include <asmjit/x86.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
using namespace asmjit;
using namespace asmjit::x86;

static const Gp R[15] = {rax, rcx, rdx, rbx, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15};
static const int UNITS = 256;

static int straight(Assembler& a) {
  int n = 0;
  for (int i = 0; i < UNITS; i++) {
    Gp d = R[i % 15], s = R[(i * 7 + 3) % 15];
    a.mov(d, imm(i * 1000)); a.add(d, s); a.sub(d, imm(5)); a.mov(s, d);
    a.add(d, imm(100000)); a.sub(s, d);
    if (i % 8 == 0) { a.mov(d, imm(int64_t(0x123456789ll) * i)); n++; }
    n += 6;
  }
  a.ret(); return n + 1;
}
static int calls(Assembler& a) {
  int n = 0; Label fn = a.new_label();
  for (int i = 0; i < UNITS; i++) {
    Gp r = R[i % 15];
    a.push(rbx); a.mov(rdi, r); a.call(fn); a.pop(rbx); a.call(r11); n += 5;
  }
  a.ret(); a.bind(fn); a.ret(); return n + 2;
}
static int loops(Assembler& a) {
  int n = 0;
  for (int i = 0; i < UNITS; i++) {
    Label top = a.new_label(), out = a.new_label();
    a.bind(top); a.cmp(rcx, rdx); a.jge(out); a.add(rax, imm(1)); a.sub(rdx, imm(1)); a.jmp(top); a.bind(out);
    n += 5;
  }
  a.ret(); return n + 1;
}
static int mem(Assembler& a) {
  int n = 0;
  for (int i = 0; i < UNITS; i++) {
    a.mov(rax, qword_ptr(rdi, rcx, 3, 16 + i)); a.mov(qword_ptr(rsp, 8), r9);
    a.lea(r10, ptr(r12, r13, 1, 300 + i * 8)); a.add(rax, qword_ptr(rbp));
    a.imul(rax, qword_ptr(r13, 8)); a.mov(qword_ptr(rdi, i * 8), imm(i)); n += 6;
  }
  a.ret(); return n + 1;
}

typedef int (*Scenario)(Assembler&);
int main(int argc, char** argv) {
  const char* names[] = {"straight", "calls", "loops", "mem"};
  Scenario fns[] = {straight, calls, loops, mem};
  int iters = argc > 1 ? atoi(argv[1]) : 2000;
  const char* dump = argc > 2 ? argv[2] : nullptr;
  for (int s = 0; s < 4; s++) {
    CodeHolder code; code.init(Environment(Arch::kX64));
    Assembler a(&code);
    if (getenv("ASMJIT_OPT_SIZE")) a.add_encoding_options(EncodingOptions::kOptimizeForSize);
    int n = 0; double best = 1e30, total = 0;
    for (int it = 0; it < iters; it++) {
      auto t0 = std::chrono::steady_clock::now();
      code.reinit();
      n = fns[s](a);
      code.flatten(); code.resolve_cross_section_fixups();
      auto t1 = std::chrono::steady_clock::now();
      double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
      total += ns; if (ns < best) best = ns;
    }
    CodeBuffer& buf = code.text_section()->buffer();
    printf("asmjit %-9s instrs=%d bytes=%zu avg_ns_per_instr=%.2f min_ns_per_instr=%.2f\n",
           names[s], n, buf.size(), total / iters / n, best / n);
    if (dump) { char p[512]; snprintf(p, sizeof p, "%s/asmjit_%s.bin", dump, names[s]);
      FILE* f = fopen(p, "wb"); fwrite(buf.data(), 1, buf.size(), f); fclose(f); }
  }
}
