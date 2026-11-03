/* pyrlyn-jit C API, fast paths (same library as pyrlyn_jit.h).
 *
 * Opcode ids: resolve a mnemonic once, then emit by id. pjit_*_emit_op does the same operand
 * validation as pjit_*_emit but skips the per-call mnemonic string lookup. Ids are valid for
 * the library build that returned them (do not hard-code or persist them).
 *
 * Pool: one executable mapping shared by many functions (functions are appended; the pool
 * maps more chunks as needed and unmaps everything on pjit_pool_free). Not thread-safe.
 */
#ifndef PYRLYN_JIT_FAST_H
#define PYRLYN_JIT_FAST_H

#include "pyrlyn_jit.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t pjit_a64_opcode(const char *mnemonic); /* -1 if unknown */
int pjit_a64_emit_op(pjit_buffer *b, uint32_t opcode, const pjit_a64_op *ops, size_t n);
int32_t pjit_x64_opcode(const char *mnemonic); /* -1 if unknown */
int pjit_x64_emit_op(pjit_buffer *b, uint32_t opcode, const pjit_x64_op *ops, size_t n);

typedef struct pjit_pool pjit_pool;
int pjit_pool_new(size_t chunk_size /* 0: 256 KiB */, pjit_pool **out);
void pjit_pool_free(pjit_pool *p);
int pjit_pool_add(pjit_pool *p, const pjit_buffer *b, const void **out);

#ifdef __cplusplus
}
#endif

#endif
