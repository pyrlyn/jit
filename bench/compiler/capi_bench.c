/* C API emit cost: the same hand-written AArch64 sum loop through pjit_a64_emit (mnemonic
 * string per instruction) and pjit_a64_emit_op (opcode ids resolved once), plus install via
 * pjit_jit_add (one mapping per function) vs pjit_pool_add (shared pool). Median of 1000. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pyrlyn_jit_fast.h"

#define CK(x) do { int _s = (x); if (_s) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, pjit_status_str(_s)); exit(1); } } while (0)
#define OPS(...) (pjit_a64_op[]){ __VA_ARGS__ }, sizeof((pjit_a64_op[]){ __VA_ARGS__ }) / sizeof(pjit_a64_op)
#define X pjit_a64_x
#define I pjit_a64_imm
enum { HS = 2 };

static uint64_t now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int cmpu(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median(uint64_t *v, int n) { qsort(v, n, sizeof *v, cmpu); return v[n / 2]; }

static uint32_t op_lsl, op_add, op_mov, op_cmp, op_ldr, op_ret;

static void emit_str(pjit_buffer *b) {
    uint32_t loop, done;
    CK(pjit_label_new(b, &loop)); CK(pjit_label_new(b, &done));
    CK(pjit_a64_emit(b, "lsl", OPS(X(2), X(1), I(3))));
    CK(pjit_a64_emit(b, "add", OPS(X(1), X(0), X(2))));
    CK(pjit_a64_emit(b, "mov", OPS(X(2), I(0))));
    CK(pjit_label_bind(b, loop));
    CK(pjit_a64_emit(b, "cmp", OPS(X(0), X(1))));
    CK(pjit_a64_branch(b, HS, done));
    CK(pjit_a64_emit(b, "ldr", OPS(X(3), pjit_a64_ptr(0, 0))));
    CK(pjit_a64_emit(b, "add", OPS(X(2), X(2), X(3))));
    CK(pjit_a64_emit(b, "add", OPS(X(0), X(0), I(8))));
    CK(pjit_a64_branch(b, -1, loop));
    CK(pjit_label_bind(b, done));
    CK(pjit_a64_emit(b, "mov", OPS(X(0), X(2))));
    CK(pjit_a64_emit(b, "ret", NULL, 0));
}
static void emit_op(pjit_buffer *b) {
    uint32_t loop, done;
    CK(pjit_label_new(b, &loop)); CK(pjit_label_new(b, &done));
    CK(pjit_a64_emit_op(b, op_lsl, OPS(X(2), X(1), I(3))));
    CK(pjit_a64_emit_op(b, op_add, OPS(X(1), X(0), X(2))));
    CK(pjit_a64_emit_op(b, op_mov, OPS(X(2), I(0))));
    CK(pjit_label_bind(b, loop));
    CK(pjit_a64_emit_op(b, op_cmp, OPS(X(0), X(1))));
    CK(pjit_a64_branch(b, HS, done));
    CK(pjit_a64_emit_op(b, op_ldr, OPS(X(3), pjit_a64_ptr(0, 0))));
    CK(pjit_a64_emit_op(b, op_add, OPS(X(2), X(2), X(3))));
    CK(pjit_a64_emit_op(b, op_add, OPS(X(0), X(0), I(8))));
    CK(pjit_a64_branch(b, -1, loop));
    CK(pjit_label_bind(b, done));
    CK(pjit_a64_emit_op(b, op_mov, OPS(X(0), X(2))));
    CK(pjit_a64_emit_op(b, op_ret, NULL, 0));
}

int main(void) {
    op_lsl = (uint32_t)pjit_a64_opcode("lsl"); op_add = (uint32_t)pjit_a64_opcode("add");
    op_mov = (uint32_t)pjit_a64_opcode("mov"); op_cmp = (uint32_t)pjit_a64_opcode("cmp");
    op_ldr = (uint32_t)pjit_a64_opcode("ldr"); op_ret = (uint32_t)pjit_a64_opcode("ret");
    pjit_buffer *b; CK(pjit_buffer_new(&b));
    enum { N = 1000 };
    static uint64_t ts[N], to[N], tr[N], tp[N];
    size_t n1 = 0, n2 = 0; const uint8_t *c;
    for (int rep = 0; rep < 2; rep++) { /* rep 0 = warm-up */
        for (int i = 0; i < N; i++) { pjit_buffer_reset(b); uint64_t t0 = now(); emit_str(b); CK(pjit_finalize(b)); ts[i] = now() - t0; }
        c = pjit_buffer_code(b, &n1);
        uint8_t ref[64]; memcpy(ref, c, n1);
        for (int i = 0; i < N; i++) { pjit_buffer_reset(b); uint64_t t0 = now(); emit_op(b); CK(pjit_finalize(b)); to[i] = now() - t0; }
        c = pjit_buffer_code(b, &n2);
        if (n1 != n2 || memcmp(ref, c, n1)) { fprintf(stderr, "bytes differ\n"); return 1; }
    }
    pjit_pool *pool; CK(pjit_pool_new(0, &pool));
    for (int i = 0; i < N; i++) { uint64_t t0 = now(); pjit_func *f; CK(pjit_jit_add(b, &f)); tr[i] = now() - t0; pjit_func_free(f); }
    const void *pf = NULL;
    for (int i = 0; i < N; i++) { uint64_t t0 = now(); CK(pjit_pool_add(pool, b, &pf)); tp[i] = now() - t0; }
    typedef int64_t (*sum_fn)(const int64_t *, int64_t);
    int64_t d[4] = {1, 2, 3, 4};
    int64_t r = ((sum_fn)pf)(d, 4);
    printf("capi a64 sum (%zu B): emit_str_ns=%llu emit_op_ns=%llu jit_add_ns=%llu pool_add_ns=%llu check=%lld\n", n1,
           (unsigned long long)median(ts, N), (unsigned long long)median(to, N), (unsigned long long)median(tr, N), (unsigned long long)median(tp, N), (long long)r);
    pjit_pool_free(pool);
    pjit_buffer_free(b);
    return r == 10 ? 0 : 1;
}
