/* pyrlyn-jit C API. Stable ABI: functions only take/return scalars, opaque handles and the
 * plain structs below. Every call validates its arguments and returns pjit_status
 * (never aborts on bad input). Link with libpyrlyn-jit (static or shared) and libc.
 * Version: pjit_version() == PJIT_VERSION (MAJOR << 16 | MINOR << 8 | PATCH). */
#ifndef PYRLYN_JIT_H
#define PYRLYN_JIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PJIT_VERSION 0x000100u

typedef enum pjit_status {
    PJIT_OK = 0,
    PJIT_ERR_OUT_OF_MEMORY,
    PJIT_ERR_INVALID_ARGUMENT,
    PJIT_ERR_UNKNOWN_MNEMONIC,
    PJIT_ERR_NO_MATCHING_FORM,
    PJIT_ERR_INVALID_LABEL,
    PJIT_ERR_LABEL_ALREADY_BOUND,
    PJIT_ERR_UNBOUND_LABEL,
    PJIT_ERR_DISPLACEMENT_OUT_OF_RANGE,
    PJIT_ERR_INVALID_FIXUP,
    PJIT_ERR_CODE_TOO_LARGE,
    PJIT_ERR_MAP_FAILED,
    PJIT_ERR_PROTECT_FAILED,
    PJIT_ERR_OTHER
} pjit_status;

typedef struct pjit_buffer pjit_buffer; /* code + labels + fixups */
typedef struct pjit_func pjit_func;     /* executable (W^X) copy of finalized code */

uint32_t pjit_version(void);
const char *pjit_status_str(int status);

/* ---- code buffer and labels ---- */
int pjit_buffer_new(pjit_buffer **out);
void pjit_buffer_free(pjit_buffer *b); /* NULL is a no-op */
void pjit_buffer_reset(pjit_buffer *b);
const uint8_t *pjit_buffer_code(const pjit_buffer *b, size_t *len);
int pjit_buffer_emit_bytes(pjit_buffer *b, const uint8_t *bytes, size_t n);
int pjit_label_new(pjit_buffer *b, uint32_t *out);
int pjit_label_bind(pjit_buffer *b, uint32_t label);
/* Resolves label fixups; on error the code is left unchanged. */
int pjit_finalize(pjit_buffer *b);

/* ---- AArch64 (all GP and ASIMD forms from AsmJit's isa_aarch64.json) ---- */
enum { PJIT_A64_GP = 0, PJIT_A64_VEC, PJIT_A64_IMM, PJIT_A64_FIMM, PJIT_A64_SHIFT, PJIT_A64_MEM, PJIT_A64_COND, PJIT_A64_REL };
enum { PJIT_A64_VB = 0, PJIT_A64_VH, PJIT_A64_VS, PJIT_A64_VD, PJIT_A64_VQ, PJIT_A64_V };
enum { PJIT_A64_ARR_NONE = 0, PJIT_A64_8B, PJIT_A64_16B, PJIT_A64_4H, PJIT_A64_8H, PJIT_A64_2S, PJIT_A64_4S, PJIT_A64_1D, PJIT_A64_2D, PJIT_A64_1Q, PJIT_A64_4B, PJIT_A64_2H, PJIT_A64_2B, PJIT_A64_1S };
enum { PJIT_A64_LSL = 0, PJIT_A64_LSR, PJIT_A64_ASR, PJIT_A64_ROR, PJIT_A64_MSL, PJIT_A64_UXTB, PJIT_A64_UXTH, PJIT_A64_UXTW, PJIT_A64_UXTX, PJIT_A64_SXTB, PJIT_A64_SXTH, PJIT_A64_SXTW, PJIT_A64_SXTX };
enum { PJIT_A64_MEM_OFF = 0, PJIT_A64_MEM_PRE, PJIT_A64_MEM_POST };
#define PJIT_A64_ZR 31 /* WZR/XZR */
#define PJIT_A64_SP 32 /* WSP/SP */

typedef struct pjit_a64_op {
    uint8_t kind;    /* PJIT_A64_* */
    uint8_t reg;     /* GP 0..32 / vector 0..31 / memory base 0..32 */
    uint8_t sub;     /* GP: 1 = X, 0 = W; VEC: PJIT_A64_V*; SHIFT: PJIT_A64_LSL..; MEM: PJIT_A64_MEM_* */
    uint8_t arr;     /* VEC arrangement (PJIT_A64_8B..) */
    uint8_t et;      /* VEC element type for elements: 0 none, 1 b, 2 h, 3 s, 4 d, 5 q */
    int8_t idx;      /* VEC element index, -1 = none */
    uint8_t index;   /* MEM index register, 0xFF = none */
    uint8_t index64; /* MEM index is X */
    uint8_t ext_op, ext_n, has_ext; /* MEM index extend/shift */
    uint8_t _pad[5];
    int64_t imm;     /* IMM value, SHIFT amount, MEM offset, COND code, REL byte offset */
    double fimm;     /* FIMM value */
} pjit_a64_op;

int pjit_a64_emit(pjit_buffer *b, const char *mnemonic, const pjit_a64_op *ops, size_t n);
/* cond < 0: `b label`; cond 0..15: `b.<cond> label` */
int pjit_a64_branch(pjit_buffer *b, int cond, uint32_t label);

static inline pjit_a64_op pjit_a64_x(uint8_t r) { pjit_a64_op o = {0}; o.kind = PJIT_A64_GP; o.reg = r; o.sub = 1; o.idx = -1; o.index = 0xFF; return o; }
static inline pjit_a64_op pjit_a64_w(uint8_t r) { pjit_a64_op o = pjit_a64_x(r); o.sub = 0; return o; }
static inline pjit_a64_op pjit_a64_imm(int64_t v) { pjit_a64_op o = pjit_a64_x(0); o.kind = PJIT_A64_IMM; o.imm = v; return o; }
static inline pjit_a64_op pjit_a64_vec(uint8_t r, uint8_t arr) { pjit_a64_op o = pjit_a64_x(r); o.kind = PJIT_A64_VEC; o.sub = PJIT_A64_V; o.arr = arr; return o; }
static inline pjit_a64_op pjit_a64_ptr(uint8_t base, int64_t off) { pjit_a64_op o = pjit_a64_x(base); o.kind = PJIT_A64_MEM; o.sub = PJIT_A64_MEM_OFF; o.imm = off; return o; }

/* ---- x86-64 (GP instruction forms) ---- */
enum { PJIT_X64_REG = 0, PJIT_X64_MEM, PJIT_X64_IMM };
enum { PJIT_RAX = 0, PJIT_RCX, PJIT_RDX, PJIT_RBX, PJIT_RSP, PJIT_RBP, PJIT_RSI, PJIT_RDI, PJIT_R8, PJIT_R9, PJIT_R10, PJIT_R11, PJIT_R12, PJIT_R13, PJIT_R14, PJIT_R15 };
#define PJIT_X64_NONE 0xFF

typedef struct pjit_x64_op {
    uint8_t kind;  /* PJIT_X64_* */
    uint8_t reg;   /* REG: PJIT_RAX.. */
    uint8_t size;  /* operand size in bytes: 1, 2, 4, 8 (0 = unsized memory) */
    uint8_t high;  /* REG: AH/CH/DH/BH */
    uint8_t base;  /* MEM: base or PJIT_X64_NONE */
    uint8_t index; /* MEM: index or PJIT_X64_NONE */
    uint8_t scale; /* MEM: 0..3 => x1, x2, x4, x8 */
    uint8_t rip;   /* MEM: [rip + disp] */
    int32_t disp;
    uint32_t _pad;
    int64_t imm;
} pjit_x64_op;

int pjit_x64_emit(pjit_buffer *b, const char *mnemonic, const pjit_x64_op *ops, size_t n);
/* cc < 0: `jmp label`; cc 0..15: `j<cc> label` (rel32) */
int pjit_x64_jump(pjit_buffer *b, int cc, uint32_t label);

static inline pjit_x64_op pjit_x64_reg(uint8_t r, uint8_t size) { pjit_x64_op o = {0}; o.kind = PJIT_X64_REG; o.reg = r; o.size = size; o.base = o.index = PJIT_X64_NONE; return o; }
static inline pjit_x64_op pjit_x64_imm(int64_t v) { pjit_x64_op o = pjit_x64_reg(0, 0); o.kind = PJIT_X64_IMM; o.imm = v; return o; }
static inline pjit_x64_op pjit_x64_mem(uint8_t base, int32_t disp, uint8_t size) { pjit_x64_op o = pjit_x64_reg(0, size); o.kind = PJIT_X64_MEM; o.base = base; o.disp = disp; return o; }

/* ---- executable memory (W^X: never writable and executable at once) ---- */
/* `b` must be finalized. */
int pjit_jit_add(const pjit_buffer *b, pjit_func **out);
const void *pjit_func_ptr(const pjit_func *f);
void pjit_func_free(pjit_func *f);

#ifdef __cplusplus
}
#endif
#endif /* PYRLYN_JIT_H */
