# pyrlyn-jit C API (and C++, Rust, Zig)

`include/pyrlyn_jit.h` is a stable C ABI over pyrlyn-jit. It uses only scalars, opaque handles
(`pjit_buffer`, `pjit_func`) and two plain operand structs (`pjit_a64_op`, `pjit_x64_op`) with
explicit padding.

## Versioning

- Header: `PJIT_VERSION_MAJOR/MINOR/PATCH`, `PJIT_VERSION` (`PJIT_MAKE_VERSION(major, minor, patch)`
  = `MAJOR << 16 | MINOR << 8 | PATCH`). Library: `pjit_version()`. Current: 0.2.0.
- Within one major version the ABI only grows: functions and enum values are appended,
  existing ones never change value, meaning or struct layout (`tests/c/capi_test.c` pins the
  struct sizes and status values with `_Static_assert`).
- `PJIT_ABI_COMPATIBLE(pjit_version())` is true when the loaded library provides everything the
  header declares (same major, minor.patch at least the header's). Check it once at startup
  when linking the shared library.

| version | added |
|---|---|
| 0.1.0 | buffers, labels, AArch64/x86-64 emitters, executable memory |
| 0.2.0 | version macros, `pjit_last_error`/`pjit_clear_error`, IR compiler (`pjit_ir_compile`, `pjit_ir_eval`), statuses `PJIT_ERR_UNSUPPORTED`..`PJIT_ERR_STEP_LIMIT` |

## Building

```sh
zig build                 # zig-out/lib/libpyrlyn-jit.a, libpyrlyn-jit.{dylib,so}, zig-out/include/
zig build lib             # static library only
zig build examples        # build and run examples/{c,cpp,zig}
zig build examples -Drust=true   # + examples/rust (needs rustc; no crates)
```

Link with `-lpyrlyn-jit` (and libc). On macOS, Apple's linker rejects Zig-built static archives
whose members are not 8-byte aligned. The build therefore also emits a plain object
(`pyrlyn_jit.o`), which the C++ and Rust examples link directly. You can also link the shared library.

## Model

1. `pjit_buffer_new` creates a code buffer.
2. Emit instructions: `pjit_a64_emit(b, "add", ops, n)` / `pjit_x64_emit(b, "mov", ops, n)`.
   Mnemonics are AsmJit's names; every AArch64 GP/ASIMD form and every x86-64 GP form is
   available.
3. Labels: `pjit_label_new`, `pjit_label_bind`, and branches `pjit_a64_branch` / `pjit_x64_jump`.
4. `pjit_finalize` resolves labels. On error the code is left unchanged.
5. Or compile IR text instead of emitting instructions: `pjit_ir_compile(b, src, len, arch)`
   with `PJIT_ARCH_HOST`, `PJIT_ARCH_X86_64`, `PJIT_ARCH_AARCH64`, `PJIT_ARCH_X86` or
   `PJIT_ARCH_RISCV64` (IR syntax: `src/core/ir.zig`). Non-host targets give bytes only.
   `pjit_ir_eval` runs the same text in the reference interpreter (for checking results).
6. `pjit_jit_add` copies the code into W^X executable memory. `pjit_func_ptr` gives the entry
   point, and `pjit_func_free` releases it.

## Errors

Every function returns a `pjit_status`; `pjit_status_str` describes it, and
`pjit_last_error()` returns the detail (internal error name, e.g. `Syntax`, `UnsupportedType`)
of the last failure on the calling thread ("" when none; `pjit_clear_error` resets it). Invalid input never
crashes. Bad registers, unknown enum values, NULL pointers, unknown mnemonics, operand sets
that no form accepts, foreign labels and out-of-range branches all return errors. The
library is fuzzed (`src/fuzz.zig`); the C API has Zig unit tests (`src/c_api.zig`) and C tests
that call every exported function, including error paths (`tests/c/capi_test.c`,
`zig build test-capi`, part of `zig build test`).

## Operands

| kind | C helper | fields |
|---|---|---|
| AArch64 GP register | `pjit_a64_x(n)`, `pjit_a64_w(n)` | `reg` 0..30, 31 = ZR, 32 = SP |
| AArch64 vector | `pjit_a64_vec(n, PJIT_A64_4S)` | `arr`, or `sub` = scalar kind; `et`/`idx` for elements |
| AArch64 immediate | `pjit_a64_imm(v)` | `imm`; `fimm` for `PJIT_A64_FIMM` |
| AArch64 memory | `pjit_a64_ptr(base, off)` | `sub` = off/pre/post, `index`, `has_ext`/`ext_op`/`ext_n` |
| x86-64 register | `pjit_x64_reg(PJIT_RAX, 8)` | `size` in bytes; `high` for AH..BH |
| x86-64 memory | `pjit_x64_mem(PJIT_RBP, -8, 8)` | `base`/`index` (`PJIT_X64_NONE`), `scale` 0..3, `rip` |
| x86-64 immediate | `pjit_x64_imm(v)` | `imm` |

## Languages

- **C**: `examples/c/main.c`.
- **C++**: `include/pyrlyn_jit.hpp` is a header-only RAII wrapper (`pyrlyn_jit::Buffer`,
  `pyrlyn_jit::Function`, and `pyrlyn_jit::Error` with a `status`). See `examples/cpp/main.cpp`.
- **Rust**: hand-written `extern "C"` declarations, with no bindgen and no crates. See
  `examples/rust/main.rs`.
All four examples also compile IR text for the host and check the result (C/C++/Rust through
`pjit_ir_compile`/`pjit_ir_eval`, Zig through `pyrlyn_jit.ir` and the host compiler).

- **Zig**: use the native API (`@import("pyrlyn_jit")`) rather than the C ABI. See
  `examples/zig/main.zig` and the README.

## Thread safety

A `pjit_buffer` must not be used from several threads at once. Different buffers and functions
are independent. On Apple Silicon, `pjit_jit_add` toggles the calling thread's JIT write
protection for the duration of the copy.
