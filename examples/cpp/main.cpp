// pyrlyn-jit C++ example (RAII wrapper include/pyrlyn_jit.hpp).
#include <cstdio>
#include <cinttypes>
#include <pyrlyn_jit.hpp>

int main() {
    try {
        pyrlyn_jit::Buffer b;
        uint32_t done = b.newLabel();
#if defined(__aarch64__)
        b.a64("add", {pjit_a64_x(0), pjit_a64_x(0), pjit_a64_x(1)});
        b.a64Branch(-1, done);
        b.a64("brk", {pjit_a64_imm(1)});
        b.bind(done);
        b.a64("ret", {});
#elif defined(__x86_64__)
#if defined(_WIN32) /* Win64 ABI: args in rcx, rdx */
#define ARG0 PJIT_RCX
#define ARG1 PJIT_RDX
#else /* System V: args in rdi, rsi */
#define ARG0 PJIT_RDI
#define ARG1 PJIT_RSI
#endif
        b.x64("mov", {pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(ARG0, 8)});
        b.x64("add", {pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(ARG1, 8)});
        b.x64Jump(-1, done);
        b.x64("int3", {});
        b.bind(done);
        b.x64("ret", {});
#endif
        b.finalize();
        try {
            b.a64("add", {pjit_a64_x(99)}); // bad register: throws, never crashes
            return 1;
        } catch (const pyrlyn_jit::Error &e) {
            if (e.status != PJIT_ERR_INVALID_ARGUMENT) return 1;
        }
        pyrlyn_jit::Function f(b);
        long r = f.as<long (*)(long, long)>()(40, 2);
        std::printf("pyrlyn_jit (C++): add(40, 2) = %ld\n", r);
        if (r != 42) return 1;

        // IR compiler: the same function from IR text, checked against the interpreter.
        pyrlyn_jit::checkAbi();
        const std::string ir = "func mul_add 3\ne:\n  v3 = mul v0 v1\n  v4 = add v3 v2\n  ret v4\n";
        b.compileIr(ir);
        pyrlyn_jit::Function g(b);
        int64_t m = g.as<int64_t (*)(int64_t, int64_t, int64_t)>()(6, 7, 0);
        std::printf("pyrlyn_jit (C++): IR mul_add(6, 7, 0) = %" PRId64 "\n", m);
        return m == 42 && pyrlyn_jit::evalIr(ir, {6, 7, 0}) == 42 ? 0 : 1;
    } catch (const pyrlyn_jit::Error &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
