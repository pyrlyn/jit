/* C-side tests of the stable C ABI (include/pyrlyn_jit.h), built and run by `zig build test`.
 * Exercises every exported function from C, including error paths. */
#include <stdio.h>
#include <string.h>
#include <pyrlyn_jit.h>

static int failures = 0;
#define EXPECT(c) do { if (!(c)) { fprintf(stderr, "%s:%d: EXPECT(%s) failed (last error: %s)\n", __FILE__, __LINE__, #c, pjit_last_error()); failures++; } } while (0)
#define EQ(a, b) EXPECT((a) == (b))

_Static_assert(PJIT_VERSION == 0x000200u, "header version");
_Static_assert(sizeof(pjit_x64_op) == 24, "pjit_x64_op layout");
_Static_assert(sizeof(pjit_a64_op) == 32, "pjit_a64_op layout");
_Static_assert(PJIT_ERR_OTHER == 13 && PJIT_ERR_STEP_LIMIT == 18, "status values are ABI");

static const char add_ir[] = "func add 2\ne:\n  v2 = add v0 v1\n  ret v2\n";
static const char fib_ir[] =
    "func fib 1\n"
    "e:\n  v1 = const 0\n  v2 = const 1\n  v3 = const 0\n  br l\n"
    "l:\n  cbr ge v3 v0 done body\n"
    "body:\n  v4 = add v1 v2\n  v1 = copy v2\n  v2 = copy v4\n  v5 = const 1\n  v3 = add v3 v5\n  br l\n"
    "done:\n  ret v1\n";

int main(void) {
    /* version */
    EXPECT(PJIT_ABI_COMPATIBLE(pjit_version()));
    EQ(pjit_version(), PJIT_VERSION);
    EXPECT(!PJIT_ABI_COMPATIBLE(PJIT_MAKE_VERSION(PJIT_VERSION_MAJOR + 1, 0, 0)));

    /* status strings: every value has one, out-of-range is handled */
    for (int s = PJIT_OK; s <= PJIT_ERR_STEP_LIMIT; s++) EXPECT(pjit_status_str(s) && *pjit_status_str(s));
    EXPECT(strcmp(pjit_status_str(1000), "unknown status") == 0);

    /* NULL handling */
    EQ(pjit_buffer_new(NULL), PJIT_ERR_INVALID_ARGUMENT);
    EXPECT(strcmp(pjit_last_error(), "InvalidArgument") == 0);
    pjit_clear_error();
    pjit_buffer_free(NULL);
    pjit_func_free(NULL);
    EXPECT(pjit_func_ptr(NULL) == NULL);
    EXPECT(pjit_buffer_code(NULL, NULL) == NULL);

    /* buffer, bytes, labels */
    pjit_buffer *b = NULL;
    EQ(pjit_buffer_new(&b), PJIT_OK);
    const uint8_t raw[] = {0x90, 0x90};
    EQ(pjit_buffer_emit_bytes(b, raw, 2), PJIT_OK);
    EQ(pjit_buffer_emit_bytes(b, NULL, 1), PJIT_ERR_INVALID_ARGUMENT);
    size_t n = 0;
    EXPECT(pjit_buffer_code(b, &n) != NULL && n == 2);
    uint32_t l;
    EQ(pjit_label_new(b, &l), PJIT_OK);
    EQ(pjit_label_bind(b, l), PJIT_OK);
    EQ(pjit_label_bind(b, l), PJIT_ERR_LABEL_ALREADY_BOUND);
    EQ(pjit_label_bind(b, 9999), PJIT_ERR_INVALID_LABEL);
    pjit_buffer_reset(b);
    pjit_buffer_code(b, &n);
    EQ(n, 0u);

    /* raw emitters: both ISAs encode on any host */
    EQ(pjit_a64_emit(b, "add", (pjit_a64_op[]){pjit_a64_x(0), pjit_a64_x(0), pjit_a64_x(1)}, 3), PJIT_OK);
    EQ(pjit_a64_emit(b, "nope", NULL, 0), PJIT_ERR_UNKNOWN_MNEMONIC);
    EXPECT(strcmp(pjit_last_error(), "UnknownMnemonic") == 0);
    EQ(pjit_x64_emit(b, "nope", NULL, 0), PJIT_ERR_UNKNOWN_MNEMONIC);
    EXPECT(strcmp(pjit_last_error(), "UnknownMnemonic") == 0);
    EQ(pjit_x64_emit(b, "mov", (pjit_x64_op[]){pjit_x64_reg(PJIT_RAX, 8), pjit_x64_imm(1)}, 2), PJIT_OK);
    EXPECT(strcmp(pjit_last_error(), "UnknownMnemonic") == 0);
    EQ(pjit_x64_emit(b, "mov", (pjit_x64_op[]){pjit_x64_reg(PJIT_RAX, 8), pjit_x64_reg(PJIT_RAX, 4)}, 2), PJIT_ERR_NO_MATCHING_FORM);
    EQ(pjit_label_new(b, &l), PJIT_OK);
    EQ(pjit_x64_jump(b, 99, l), PJIT_ERR_INVALID_ARGUMENT);
    EQ(pjit_x64_jump(b, -1, l), PJIT_OK);
    EQ(pjit_finalize(b), PJIT_ERR_UNBOUND_LABEL);
    pjit_func *f = NULL;
    EQ(pjit_jit_add(b, &f), PJIT_ERR_UNBOUND_LABEL); /* not finalized */
    pjit_buffer_reset(b);

    /* IR: interpreter */
    int64_t r = 0;
    EQ(pjit_ir_eval(add_ir, sizeof add_ir - 1, (int64_t[]){40, 2}, 2, 0, &r), PJIT_OK);
    EQ(r, 42);
    EQ(pjit_ir_eval(fib_ir, sizeof fib_ir - 1, (int64_t[]){50}, 1, 0, &r), PJIT_OK);
    EQ(r, 12586269025LL);
    EQ(pjit_ir_eval(fib_ir, sizeof fib_ir - 1, (int64_t[]){1000000}, 1, 100, &r), PJIT_ERR_STEP_LIMIT);
    EQ(pjit_ir_eval("func f 1\ne:\n  v1 = frob v0\n", 26, (int64_t[]){1}, 1, 0, &r), PJIT_ERR_IR_PARSE);
    EXPECT(strcmp(pjit_last_error(), "Syntax") == 0);
    pjit_clear_error();
    EXPECT(*pjit_last_error() == 0);
    EQ(pjit_ir_eval("func f 1\ne:\n  v1 = copy v0\n", 26, (int64_t[]){1}, 1, 0, &r), PJIT_ERR_IR_VERIFY);

    const char bad_type[] = "func f 0\ne:\n  v0 = fconst.f64 1\n  ret v0\n";
    r = 123;
    EQ(pjit_ir_eval(bad_type, sizeof bad_type - 1, NULL, 0, 0, &r), PJIT_ERR_IR_TYPE);
    EQ(r, 123); /* errors leave the output alone */
    const char bad_lane[] = "func f 1\ne:\n  v1 = splat.v2i64 v0\n  v2 = extract.v2i64 v1 2\n  ret v2\n";
    EQ(pjit_ir_eval(bad_lane, sizeof bad_lane - 1, (int64_t[]){1}, 1, 0, &r), PJIT_ERR_IR_TYPE);
    EXPECT(strcmp(pjit_last_error(), "TypeMismatch") == 0);
    EQ(pjit_ir_eval(add_ir, sizeof add_ir - 1, NULL, 2, 0, &r), PJIT_ERR_INVALID_ARGUMENT);
    EXPECT(strcmp(pjit_last_error(), "InvalidArgument") == 0);
    EQ(pjit_ir_eval(add_ir, sizeof add_ir - 1, (int64_t[]){40, 2}, 2, UINT64_MAX, &r),
       SIZE_MAX < UINT64_MAX ? PJIT_ERR_INVALID_ARGUMENT : PJIT_OK);

    /* AArch64 branch entry point, including a valid finalized branch. */
    EQ(pjit_label_new(b, &l), PJIT_OK);
    EQ(pjit_a64_branch(b, -1, l), PJIT_OK);
    EQ(pjit_label_bind(b, l), PJIT_OK);
    EQ(pjit_finalize(b), PJIT_OK);
    pjit_buffer_reset(b);

    /* IR: compile for every arch (bytes only), run on the host */
    for (int a = PJIT_ARCH_X86_64; a <= PJIT_ARCH_RISCV64; a++) {
        pjit_buffer_reset(b);
        EQ(pjit_ir_compile(b, fib_ir, sizeof fib_ir - 1, a), PJIT_OK);
        pjit_buffer_code(b, &n);
        EXPECT(n > 0);
    }
    EQ(pjit_ir_compile(b, fib_ir, sizeof fib_ir - 1, 42), PJIT_ERR_INVALID_ARGUMENT);
    pjit_buffer_reset(b);
    EQ(pjit_ir_compile(b, fib_ir, sizeof fib_ir - 1, PJIT_ARCH_HOST), PJIT_OK);
    EQ(pjit_jit_add(b, &f), PJIT_OK);
#if defined(__i386__) || defined(_M_IX86)
    /* The x86 backend uses 32-bit values and cdecl. Keep Fibonacci in int32_t range. */
    int32_t (*fib)(int32_t) = (int32_t (*)(int32_t))pjit_func_ptr(f);
    const int64_t max_k = 45;
#else
    int64_t (*fib)(int64_t) = (int64_t (*)(int64_t))pjit_func_ptr(f);
    const int64_t max_k = 90;
#endif
    for (int64_t k = 0; k <= max_k; k += 15) {
        int64_t want;
        EQ(pjit_ir_eval(fib_ir, sizeof fib_ir - 1, &k, 1, 0, &want), PJIT_OK);
        EQ(fib(k), want);
    }
    pjit_func_free(f);
    pjit_buffer_free(b);

    if (failures) { fprintf(stderr, "capi_test: %d failure(s)\n", failures); return 1; }
    printf("capi_test: ok (pyrlyn_jit %u.%u.%u)\n", pjit_version() >> 16, (pjit_version() >> 8) & 0xFF, pjit_version() & 0xFF);
    return 0;
}
