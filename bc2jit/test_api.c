#include "bc2jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "C API check failed at %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    uint8_t bytes[] = {'K','X','B','C',0,2,6,0,2,0,2,1,0x10,0xf0};
    pjit_bc2jit *h = NULL;
    pjit_bc2jit_diagnostic d = {99, "old"};
    size_t n = 99;
    CHECK(pjit_bc2jit_code(NULL, &n) == NULL && n == 0);
    CHECK(pjit_bc2jit_entry(NULL) == NULL);
    pjit_bc2jit_free(NULL);
    CHECK(pjit_bc2jit_compile(bytes, sizeof bytes, 0, 0, NULL, &d) == PJIT_BC2JIT_INVALID_ARGUMENT);
    CHECK(d.offset == 0 && d.message == NULL);
    if (!pjit_bc2jit_enabled()) {
        CHECK(pjit_bc2jit_compile(bytes, sizeof bytes, 0, 0, &h, &d) == PJIT_BC2JIT_DISABLED);
        CHECK(h == NULL && d.offset == 0 && d.message == NULL);
        return 0;
    }
    CHECK(pjit_bc2jit_compile(NULL, 1, PJIT_BC2JIT_RISCV64, 0, &h, &d) == PJIT_BC2JIT_INVALID_ARGUMENT);
    CHECK(pjit_bc2jit_compile(bytes, sizeof bytes, 99, 0, &h, NULL) == PJIT_BC2JIT_INVALID_ARGUMENT);
    CHECK(pjit_bc2jit_compile(bytes, sizeof bytes, 0, 2, &h, NULL) == PJIT_BC2JIT_INVALID_ARGUMENT);
    CHECK(pjit_bc2jit_compile(NULL, 0, PJIT_BC2JIT_RISCV64, 0, &h, &d) == PJIT_BC2JIT_INVALID_BYTECODE);
    CHECK(h == NULL && d.offset == 0 && strcmp(d.message, "Truncated") == 0);
    bytes[8] = 0x99;
    CHECK(pjit_bc2jit_compile(bytes, sizeof bytes, PJIT_BC2JIT_RISCV64, 0, &h, &d) == PJIT_BC2JIT_INVALID_BYTECODE);
    CHECK(h == NULL && d.offset == 8 && strcmp(d.message, "UnknownOpcode") == 0);
    bytes[8] = 2;
    for (int version = 0; version <= 1; ++version) {
        bytes[4] = (uint8_t)version;
        for (int target = 1; target <= 3; ++target) {
            int status = pjit_bc2jit_compile(bytes, sizeof bytes, target, PJIT_BC2JIT_NO_JIT, &h, &d);
            if (status == PJIT_BC2JIT_BACKEND_NOT_BUILT) { CHECK(h == NULL); continue; }
            CHECK(status == PJIT_BC2JIT_OK);
            CHECK(pjit_bc2jit_code(h, &n) != NULL && n > 0);
            CHECK(pjit_bc2jit_entry(h) == NULL);
            CHECK(d.message == NULL && d.offset == 0);
            pjit_bc2jit_free(h);
            h = NULL;
        }
        int status = pjit_bc2jit_compile(bytes, sizeof bytes, 0, 0, &h, &d);
        if (status == PJIT_BC2JIT_BACKEND_NOT_BUILT || status == PJIT_BC2JIT_UNSUPPORTED_HOST) continue;
        CHECK(status == PJIT_BC2JIT_OK);
        pjit_bc2jit_fn f = pjit_bc2jit_entry(h);
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
        if (version == 0) CHECK(f == NULL); else
#endif
        { CHECK(f != NULL); CHECK(f(40, 2, 0, 0) == 42); CHECK(f(-9, 5, 0, 0) == -4); }
        pjit_bc2jit_free(h);
        h = NULL;
    }
    return 0;
}
