// ZigJIT C++ example (RAII wrapper include/zigjit.hpp).
#include <cstdio>
#include <zigjit.hpp>

int main() {
    try {
        zigjit::Buffer b;
        uint32_t done = b.newLabel();
#if defined(__aarch64__)
        b.a64("add", {zj_a64_x(0), zj_a64_x(0), zj_a64_x(1)});
        b.a64Branch(-1, done);
        b.a64("brk", {zj_a64_imm(1)});
        b.bind(done);
        b.a64("ret", {});
#elif defined(__x86_64__)
        b.x64("mov", {zj_x64_reg(ZJ_RAX, 8), zj_x64_reg(ZJ_RDI, 8)});
        b.x64("add", {zj_x64_reg(ZJ_RAX, 8), zj_x64_reg(ZJ_RSI, 8)});
        b.x64Jump(-1, done);
        b.x64("int3", {});
        b.bind(done);
        b.x64("ret", {});
#endif
        b.finalize();
        try {
            b.a64("add", {zj_a64_x(99)}); // bad register: throws, never crashes
            return 1;
        } catch (const zigjit::Error &e) {
            if (e.status != ZJ_ERR_INVALID_ARGUMENT) return 1;
        }
        zigjit::Function f(b);
        long r = f.as<long (*)(long, long)>()(40, 2);
        std::printf("zigjit (C++): add(40, 2) = %ld\n", r);
        return r == 42 ? 0 : 1;
    } catch (const zigjit::Error &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
