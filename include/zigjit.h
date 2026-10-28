/* ZigJIT C API. Stable ABI: functions only take/return scalars, opaque handles and the
 * plain structs below. Every call validates its arguments and returns zj_status
 * (never aborts on bad input). Link with libzigjit (static or shared) and libc.
 * Version: zj_version() == ZJ_VERSION (MAJOR << 16 | MINOR << 8 | PATCH). */
#ifndef ZIGJIT_H
#define ZIGJIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZJ_VERSION 0x000100u

typedef enum zj_status {
    ZJ_OK = 0,
    ZJ_ERR_OUT_OF_MEMORY,
    ZJ_ERR_INVALID_ARGUMENT,
    ZJ_ERR_UNKNOWN_MNEMONIC,
    ZJ_ERR_NO_MATCHING_FORM,
    ZJ_ERR_INVALID_LABEL,
    ZJ_ERR_LABEL_ALREADY_BOUND,
    ZJ_ERR_UNBOUND_LABEL,
    ZJ_ERR_DISPLACEMENT_OUT_OF_RANGE,
    ZJ_ERR_INVALID_FIXUP,
    ZJ_ERR_CODE_TOO_LARGE,
    ZJ_ERR_MAP_FAILED,
    ZJ_ERR_PROTECT_FAILED,
    ZJ_ERR_OTHER
} zj_status;

typedef struct zj_buffer zj_buffer; /* code + labels + fixups */
typedef struct zj_func zj_func;     /* executable (W^X) copy of finalized code */

uint32_t zj_version(void);
const char *zj_status_str(int status);

/* ---- code buffer and labels ---- */
int zj_buffer_new(zj_buffer **out);
void zj_buffer_free(zj_buffer *b); /* NULL is a no-op */
void zj_buffer_reset(zj_buffer *b);
const uint8_t *zj_buffer_code(const zj_buffer *b, size_t *len);
int zj_buffer_emit_bytes(zj_buffer *b, const uint8_t *bytes, size_t n);
int zj_label_new(zj_buffer *b, uint32_t *out);
int zj_label_bind(zj_buffer *b, uint32_t label);
/* Resolves label fixups; on error the code is left unchanged. */
int zj_finalize(zj_buffer *b);

/* ---- AArch64 (all GP and ASIMD forms from AsmJit's isa_aarch64.json) ---- */
enum { ZJ_A64_GP = 0, ZJ_A64_VEC, ZJ_A64_IMM, ZJ_A64_FIMM, ZJ_A64_SHIFT, ZJ_A64_MEM, ZJ_A64_COND, ZJ_A64_REL };
enum { ZJ_A64_VB = 0, ZJ_A64_VH, ZJ_A64_VS, ZJ_A64_VD, ZJ_A64_VQ, ZJ_A64_V };
enum { ZJ_A64_ARR_NONE = 0, ZJ_A64_8B, ZJ_A64_16B, ZJ_A64_4H, ZJ_A64_8H, ZJ_A64_2S, ZJ_A64_4S, ZJ_A64_1D, ZJ_A64_2D, ZJ_A64_1Q, ZJ_A64_4B, ZJ_A64_2H, ZJ_A64_2B, ZJ_A64_1S };
enum { ZJ_A64_LSL = 0, ZJ_A64_LSR, ZJ_A64_ASR, ZJ_A64_ROR, ZJ_A64_MSL, ZJ_A64_UXTB, ZJ_A64_UXTH, ZJ_A64_UXTW, ZJ_A64_UXTX, ZJ_A64_SXTB, ZJ_A64_SXTH, ZJ_A64_SXTW, ZJ_A64_SXTX };
enum { ZJ_A64_MEM_OFF = 0, ZJ_A64_MEM_PRE, ZJ_A64_MEM_POST };
#define ZJ_A64_ZR 31 /* WZR/XZR */
#define ZJ_A64_SP 32 /* WSP/SP */

typedef struct zj_a64_op {
    uint8_t kind;    /* ZJ_A64_* */
    uint8_t reg;     /* GP 0..32 / vector 0..31 / memory base 0..32 */
    uint8_t sub;     /* GP: 1 = X, 0 = W; VEC: ZJ_A64_V*; SHIFT: ZJ_A64_LSL..; MEM: ZJ_A64_MEM_* */
    uint8_t arr;     /* VEC arrangement (ZJ_A64_8B..) */
    uint8_t et;      /* VEC element type for elements: 0 none, 1 b, 2 h, 3 s, 4 d, 5 q */
    int8_t idx;      /* VEC element index, -1 = none */
    uint8_t index;   /* MEM index register, 0xFF = none */
    uint8_t index64; /* MEM index is X */
    uint8_t ext_op, ext_n, has_ext; /* MEM index extend/shift */
    uint8_t _pad[5];
    int64_t imm;     /* IMM value, SHIFT amount, MEM offset, COND code, REL byte offset */
    double fimm;     /* FIMM value */
} zj_a64_op;

int zj_a64_emit(zj_buffer *b, const char *mnemonic, const zj_a64_op *ops, size_t n);
/* cond < 0: `b label`; cond 0..15: `b.<cond> label` */
int zj_a64_branch(zj_buffer *b, int cond, uint32_t label);

static inline zj_a64_op zj_a64_x(uint8_t r) { zj_a64_op o = {0}; o.kind = ZJ_A64_GP; o.reg = r; o.sub = 1; o.idx = -1; o.index = 0xFF; return o; }
static inline zj_a64_op zj_a64_w(uint8_t r) { zj_a64_op o = zj_a64_x(r); o.sub = 0; return o; }
static inline zj_a64_op zj_a64_imm(int64_t v) { zj_a64_op o = zj_a64_x(0); o.kind = ZJ_A64_IMM; o.imm = v; return o; }
static inline zj_a64_op zj_a64_vec(uint8_t r, uint8_t arr) { zj_a64_op o = zj_a64_x(r); o.kind = ZJ_A64_VEC; o.sub = ZJ_A64_V; o.arr = arr; return o; }
static inline zj_a64_op zj_a64_ptr(uint8_t base, int64_t off) { zj_a64_op o = zj_a64_x(base); o.kind = ZJ_A64_MEM; o.sub = ZJ_A64_MEM_OFF; o.imm = off; return o; }

/* ---- x86-64 (GP instruction forms) ---- */
enum { ZJ_X64_REG = 0, ZJ_X64_MEM, ZJ_X64_IMM };
enum { ZJ_RAX = 0, ZJ_RCX, ZJ_RDX, ZJ_RBX, ZJ_RSP, ZJ_RBP, ZJ_RSI, ZJ_RDI, ZJ_R8, ZJ_R9, ZJ_R10, ZJ_R11, ZJ_R12, ZJ_R13, ZJ_R14, ZJ_R15 };
#define ZJ_X64_NONE 0xFF

typedef struct zj_x64_op {
    uint8_t kind;  /* ZJ_X64_* */
    uint8_t reg;   /* REG: ZJ_RAX.. */
    uint8_t size;  /* operand size in bytes: 1, 2, 4, 8 (0 = unsized memory) */
    uint8_t high;  /* REG: AH/CH/DH/BH */
    uint8_t base;  /* MEM: base or ZJ_X64_NONE */
    uint8_t index; /* MEM: index or ZJ_X64_NONE */
    uint8_t scale; /* MEM: 0..3 => x1, x2, x4, x8 */
    uint8_t rip;   /* MEM: [rip + disp] */
    int32_t disp;
    uint32_t _pad;
    int64_t imm;
} zj_x64_op;

int zj_x64_emit(zj_buffer *b, const char *mnemonic, const zj_x64_op *ops, size_t n);
/* cc < 0: `jmp label`; cc 0..15: `j<cc> label` (rel32) */
int zj_x64_jump(zj_buffer *b, int cc, uint32_t label);

static inline zj_x64_op zj_x64_reg(uint8_t r, uint8_t size) { zj_x64_op o = {0}; o.kind = ZJ_X64_REG; o.reg = r; o.size = size; o.base = o.index = ZJ_X64_NONE; return o; }
static inline zj_x64_op zj_x64_imm(int64_t v) { zj_x64_op o = zj_x64_reg(0, 0); o.kind = ZJ_X64_IMM; o.imm = v; return o; }
static inline zj_x64_op zj_x64_mem(uint8_t base, int32_t disp, uint8_t size) { zj_x64_op o = zj_x64_reg(0, size); o.kind = ZJ_X64_MEM; o.base = base; o.disp = disp; return o; }

/* ---- executable memory (W^X: never writable and executable at once) ---- */
/* `b` must be finalized. */
int zj_jit_add(const zj_buffer *b, zj_func **out);
const void *zj_func_ptr(const zj_func *f);
void zj_func_free(zj_func *f);

#ifdef __cplusplus
}
#endif
#endif /* ZIGJIT_H */
