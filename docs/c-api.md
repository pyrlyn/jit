# ZigJIT C API (and C++, Rust, Zig)

`include/zigjit.h` is a stable C ABI over ZigJIT. It uses only scalars, opaque handles
(`zj_buffer`, `zj_func`) and two plain operand structs (`zj_a64_op`, `zj_x64_op`) with
explicit padding. Version: `zj_version()` returns `MAJOR << 16 | MINOR << 8 | PATCH`
(`ZJ_VERSION`). Within a major version, functions and struct layouts are only added to,
never changed.

## Building

```sh
zig build                 # zig-out/lib/libzigjit.a, libzigjit.{dylib,so}, zig-out/include/
zig build lib             # static library only
zig build examples        # build and run examples/{c,cpp,zig}
zig build examples -Drust=true   # + examples/rust (needs rustc; no crates)
```

Link with `-lzigjit` (and libc). On macOS, Apple's linker rejects Zig-built static archives
whose members are not 8-byte aligned. The build therefore also emits a plain object
(`zigjit.o`), which the C++ and Rust examples link directly. You can also link the shared library.

## Model

1. `zj_buffer_new` creates a code buffer.
2. Emit instructions: `zj_a64_emit(b, "add", ops, n)` / `zj_x64_emit(b, "mov", ops, n)`.
   Mnemonics are AsmJit's names; every AArch64 GP/ASIMD form and every x86-64 GP form is
   available.
3. Labels: `zj_label_new`, `zj_label_bind`, and branches `zj_a64_branch` / `zj_x64_jump`.
4. `zj_finalize` resolves labels. On error the code is left unchanged.
5. `zj_jit_add` copies the code into W^X executable memory. `zj_func_ptr` gives the entry
   point, and `zj_func_free` releases it.

## Errors

Every function returns a `zj_status`; `zj_status_str` describes it. Invalid input never
crashes. Bad registers, unknown enum values, NULL pointers, unknown mnemonics, operand sets
that no form accepts, foreign labels and out-of-range branches all return errors. The
library is fuzzed (`src/fuzz.zig`) and the C API has unit tests (`src/c_api.zig`).

## Operands

| kind | C helper | fields |
|---|---|---|
| AArch64 GP register | `zj_a64_x(n)`, `zj_a64_w(n)` | `reg` 0..30, 31 = ZR, 32 = SP |
| AArch64 vector | `zj_a64_vec(n, ZJ_A64_4S)` | `arr`, or `sub` = scalar kind; `et`/`idx` for elements |
| AArch64 immediate | `zj_a64_imm(v)` | `imm`; `fimm` for `ZJ_A64_FIMM` |
| AArch64 memory | `zj_a64_ptr(base, off)` | `sub` = off/pre/post, `index`, `has_ext`/`ext_op`/`ext_n` |
| x86-64 register | `zj_x64_reg(ZJ_RAX, 8)` | `size` in bytes; `high` for AH..BH |
| x86-64 memory | `zj_x64_mem(ZJ_RBP, -8, 8)` | `base`/`index` (`ZJ_X64_NONE`), `scale` 0..3, `rip` |
| x86-64 immediate | `zj_x64_imm(v)` | `imm` |

## Languages

- **C**: `examples/c/main.c`.
- **C++**: `include/zigjit.hpp` is a header-only RAII wrapper (`zigjit::Buffer`,
  `zigjit::Function`, and `zigjit::Error` with a `status`). See `examples/cpp/main.cpp`.
- **Rust**: hand-written `extern "C"` declarations, with no bindgen and no crates. See
  `examples/rust/main.rs`.
- **Zig**: use the native API (`@import("zigjit")`) rather than the C ABI. See
  `examples/zig/main.zig` and the README.

## Thread safety

A `zj_buffer` must not be used from several threads at once. Different buffers and functions
are independent. On Apple Silicon, `zj_jit_add` toggles the calling thread's JIT write
protection for the duration of the copy.
