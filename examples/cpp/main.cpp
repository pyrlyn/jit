// pyrlyn-jit C++ example (RAII wrapper include/pyrlyn_jit.hpp).
#include <cstdio>
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
        b.x64("mov", {pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(PJIT_RDI, 8)});
        b.x64("add", {pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(PJIT_RSI, 8)});
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
        return r == 42 ? 0 : 1;
    } catch (const pyrlyn_jit::Error &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
