/* KXBC v0/v1 API. Link libpyrlyn-jit and libc. */
#ifndef PYRLYN_BC2JIT_H
#define PYRLYN_BC2JIT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct pjit_bc2jit pjit_bc2jit;
typedef int64_t (*pjit_bc2jit_fn)(int64_t, int64_t, int64_t, int64_t);
typedef enum pjit_bc2jit_status {
    PJIT_BC2JIT_OK = 0,
    PJIT_BC2JIT_INVALID_ARGUMENT,
    PJIT_BC2JIT_DISABLED,
    PJIT_BC2JIT_INVALID_BYTECODE,
    PJIT_BC2JIT_BACKEND_NOT_BUILT,
    PJIT_BC2JIT_OUT_OF_MEMORY,
    PJIT_BC2JIT_COMPILE_FAILED,
    PJIT_BC2JIT_MAP_FAILED,
    PJIT_BC2JIT_UNSUPPORTED_HOST
} pjit_bc2jit_status;
typedef enum pjit_bc2jit_target {
    PJIT_BC2JIT_HOST = 0,
    PJIT_BC2JIT_X86_64 = 1,
    PJIT_BC2JIT_AARCH64 = 2,
    PJIT_BC2JIT_RISCV64 = 3
} pjit_bc2jit_target;
#define PJIT_BC2JIT_NO_JIT 1u

typedef struct pjit_bc2jit_diagnostic {
    size_t offset;        /* Module byte offset for decoder errors; otherwise zero. */
    const char *message;  /* Static error name, NULL if unavailable. Do not free. */
} pjit_bc2jit_diagnostic;

/* Always available, including -Dbc2jit=false builds (returns 0). */
int pjit_bc2jit_enabled(void);
/* Validates the entire module before lowering. flags=0 maps executable code only for
 * a compatible host target. NO_JIT always emits code only. No interpreter fallback.
 * out is required and cleared on failure; diagnostic is optional and reset on entry.
 * Input is borrowed only during this call. NULL bytes is valid only with len=0.
 * Disabled builds return DISABLED, keeping these symbols linkable. */
int pjit_bc2jit_compile(const uint8_t *bytes, size_t len, int target, uint32_t flags,
                       pjit_bc2jit **out, pjit_bc2jit_diagnostic *diagnostic);
/* Borrowed machine-code bytes; valid until free. NULL handle returns NULL/length 0. */
const uint8_t *pjit_bc2jit_code(const pjit_bc2jit *handle, size_t *len);
/* NULL for NO_JIT, foreign targets, incompatible ABI, disabled builds or NULL handle.
 * Otherwise callable with four i64 arguments; unused arguments are ignored.
 * Do not call after free; synchronize calls against destruction. */
pjit_bc2jit_fn pjit_bc2jit_entry(const pjit_bc2jit *handle);
/* Releases code and executable mapping. NULL is a no-op. */
void pjit_bc2jit_free(pjit_bc2jit *handle);
#ifdef __cplusplus
}
#endif
#endif
