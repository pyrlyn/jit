// pyrlyn-jit C example: JIT-compiles `long add(long a, long b)` for the host and calls it.
#include <stdio.h>
#include <pyrlyn_jit.h>

#define CHECK(x) do { int s_ = (x); if (s_ != PJIT_OK) { fprintf(stderr, "%s: %s\n", #x, pjit_status_str(s_)); return 1; } } while (0)

int main(void) {
    pjit_buffer *b;
    CHECK(pjit_buffer_new(&b));
    uint32_t done;
    CHECK(pjit_label_new(b, &done));
#if defined(__aarch64__)
    pjit_a64_op add[] = { pjit_a64_x(0), pjit_a64_x(0), pjit_a64_x(1) };
    CHECK(pjit_a64_emit(b, "add", add, 3));
    CHECK(pjit_a64_branch(b, -1, done));            /* b done */
    CHECK(pjit_a64_emit(b, "brk", (pjit_a64_op[]){ pjit_a64_imm(1) }, 1)); /* skipped */
    CHECK(pjit_label_bind(b, done));
    CHECK(pjit_a64_emit(b, "ret", NULL, 0));
#elif defined(__x86_64__)
#if defined(_WIN32) /* Win64 ABI: args in rcx, rdx */
#define ARG0 PJIT_RCX
#define ARG1 PJIT_RDX
#else /* System V: args in rdi, rsi */
#define ARG0 PJIT_RDI
#define ARG1 PJIT_RSI
#endif
    pjit_x64_op mov[] = { pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(ARG0, 8) };
    pjit_x64_op add[] = { pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(ARG1, 8) };
    CHECK(pjit_x64_emit(b, "mov", mov, 2));
    CHECK(pjit_x64_emit(b, "add", add, 2));
    CHECK(pjit_x64_jump(b, -1, done));              /* jmp done */
    CHECK(pjit_x64_emit(b, "int3", NULL, 0));       /* skipped */
    CHECK(pjit_label_bind(b, done));
    CHECK(pjit_x64_emit(b, "ret", NULL, 0));
#else
#error "unsupported host"
#endif
    CHECK(pjit_finalize(b));

    /* Errors are values, not crashes: */
    if (pjit_a64_emit(b, "no_such_insn", NULL, 0) != PJIT_ERR_UNKNOWN_MNEMONIC) return 1;

    pjit_func *f;
    CHECK(pjit_jit_add(b, &f));
    long (*fn)(long, long) = (long (*)(long, long))pjit_func_ptr(f);
    long r = fn(40, 2);
    printf("pyrlyn_jit %u.%u.%u (C): add(40, 2) = %ld\n", pjit_version() >> 16, (pjit_version() >> 8) & 0xFF, pjit_version() & 0xFF, r);
    pjit_func_free(f);
    pjit_buffer_free(b);
    return r == 42 ? 0 : 1;
}
