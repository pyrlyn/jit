/* Same add(a,b) workload through the assembler C API and KXBC API.
 * Build: zig build bench-bc2jit-api -Doptimize=ReleaseFast [-Dbc2jit=false]
 * Run: zig-out/bin/bench_bc2jit_api
 * Reports median ns/op (200 warmup, 15 samples, 500 iterations; calls: 100000).
 */
#define _POSIX_C_SOURCE 200809L
#include "pyrlyn_jit.h"
#include "bc2jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "benchmark check failed at line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static volatile int64_t sink;
static uint8_t bytecode[] = {'K','X','B','C',0,2,6,0,2,0,2,1,0x10,0xf0};
static uint64_t now(void) {
#ifdef _WIN32
    LARGE_INTEGER ticks, frequency;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&frequency);
    return (uint64_t)((double)ticks.QuadPart * 1e9 / (double)frequency.QuadPart);
#else
    struct timespec ts;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}
static pjit_buffer *emit(void) {
    pjit_buffer *b = NULL;
    CHECK(pjit_buffer_new(&b) == PJIT_OK);
#if defined(__aarch64__)
    pjit_a64_op mov0[] = {pjit_a64_x(9), pjit_a64_x(0)};
    pjit_a64_op mov1[] = {pjit_a64_x(10), pjit_a64_x(1)};
    pjit_a64_op add[] = {pjit_a64_x(9), pjit_a64_x(9), pjit_a64_x(10)};
    pjit_a64_op ret[] = {pjit_a64_x(0), pjit_a64_x(9)};
    CHECK(pjit_a64_emit(b, "mov", mov0, 2) == PJIT_OK);
    CHECK(pjit_a64_emit(b, "mov", mov1, 2) == PJIT_OK);
    CHECK(pjit_a64_emit(b, "add", add, 3) == PJIT_OK);
    CHECK(pjit_a64_emit(b, "mov", ret, 2) == PJIT_OK);
    CHECK(pjit_a64_emit(b, "ret", NULL, 0) == PJIT_OK);
#elif defined(__x86_64__) && !defined(_WIN32)
    pjit_x64_op mov0[] = {pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(PJIT_RDI, 8)};
    pjit_x64_op mov1[] = {pjit_x64_reg(PJIT_R10, 8), pjit_x64_reg(PJIT_RSI, 8)};
    pjit_x64_op add[] = {pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(PJIT_R10, 8)};
    CHECK(pjit_x64_emit(b, "mov", mov0, 2) == PJIT_OK);
    CHECK(pjit_x64_emit(b, "mov", mov1, 2) == PJIT_OK);
    CHECK(pjit_x64_emit(b, "add", add, 2) == PJIT_OK);
    CHECK(pjit_x64_emit(b, "ret", NULL, 0) == PJIT_OK);
#else
    fprintf(stderr, "benchmark requires AArch64 or x86-64 SysV\n");
    exit(1);
#endif
    CHECK(pjit_finalize(b) == PJIT_OK);
    return b;
}
static void direct_compile(void) {
    pjit_buffer *b = emit();
    pjit_buffer_free(b);
}
static void direct_jit(void) {
    pjit_buffer *b = emit();
    pjit_func *h = NULL;
    CHECK(pjit_jit_add(b, &h) == PJIT_OK);
    pjit_bc2jit_fn f = (pjit_bc2jit_fn)pjit_func_ptr(h);
    CHECK(f(40, 2, 0, 0) == 42);
    pjit_func_free(h);
    pjit_buffer_free(b);
}
static void bc_run(unsigned flags) {
    pjit_bc2jit *h = NULL;
    CHECK(pjit_bc2jit_compile(bytecode, sizeof bytecode, PJIT_BC2JIT_HOST, flags, &h, NULL) == PJIT_BC2JIT_OK);
    if (!flags) { pjit_bc2jit_fn f = pjit_bc2jit_entry(h); CHECK(f && f(40, 2, 0, 0) == 42); }
    pjit_bc2jit_free(h);
}
static void bc_compile(void) { bc_run(PJIT_BC2JIT_NO_JIT); }
static void bc_jit(void) { bc_run(0); }
static pjit_bc2jit_fn cached;
static void call_cached(void) { sink = cached(40, 2, 0, 0); }
static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void measure(const char *name, void (*body)(void), int batch) {
    for (int i = 0; i < 200; ++i) body();
    double samples[15];
    for (int i = 0; i < 15; ++i) {
        uint64_t start = now();
        for (int j = 0; j < batch; ++j) body();
        samples[i] = (double)(now() - start) / batch;
    }
    qsort(samples, 15, sizeof *samples, compare);
    printf("%s %.3f\n", name, samples[7]);
}
int main(void) {
    printf("bc2jit_enabled %d\n", pjit_bc2jit_enabled());
    measure("direct_compile_ns", direct_compile, 500);
    measure("direct_jit_ns", direct_jit, 500);
    pjit_buffer *b = emit();
    pjit_func *h = NULL;
    CHECK(pjit_jit_add(b, &h) == PJIT_OK);
    cached = (pjit_bc2jit_fn)pjit_func_ptr(h);
    measure("direct_call_ns", call_cached, 100000);
    CHECK(sink == 42);
    if (pjit_bc2jit_enabled()) {
        for (int v = 0; v <= 1; ++v) {
            bytecode[4] = (uint8_t)v;
            measure(v ? "bc_v1_compile_ns" : "bc_v0_compile_ns", bc_compile, 500);
            measure(v ? "bc_v1_jit_ns" : "bc_v0_jit_ns", bc_jit, 500);
            pjit_bc2jit *bc = NULL;
            CHECK(pjit_bc2jit_compile(bytecode, sizeof bytecode, 0, 0, &bc, NULL) == PJIT_BC2JIT_OK);
            size_t nb = 0, nd = 0;
            const uint8_t *cb = pjit_bc2jit_code(bc, &nb);
            const uint8_t *cd = pjit_buffer_code(b, &nd);
            printf("bc_v%d_code_bytes %zu\n", v, nb);
            if (!v) CHECK(nb == nd && memcmp(cb, cd, nb) == 0);
            cached = pjit_bc2jit_entry(bc);
            CHECK(cached != NULL);
            measure(v ? "bc_v1_call_ns" : "bc_v0_call_ns", call_cached, 100000);
            CHECK(sink == 42);
            pjit_bc2jit_free(bc);
        }
    }
    pjit_func_free(h);
    pjit_buffer_free(b);
    return 0;
}
