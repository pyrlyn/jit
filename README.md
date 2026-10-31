# pyrlyn-jit

A small AsmJit-style machine code generator in Zig 0.17 (stdlib only).

- Backends are separate modules, picked at build time: **x86-64** and **AArch64**
  (32-bit x86, RISC-V, MIPS: see [TODO-ARCHES.md](TODO-ARCHES.md)).
- Host and target are independent. You can emit x86-64 on an arm64 Mac; running the code
  needs target == host (x86-64 runs under Rosetta 2).
- No IR, no vtables, no runtime dispatch. Each method encodes one instruction directly into
  the buffer.
- Optional **safety layer**: operand validation and emission hooks. Both are chosen at
  comptime and cost nothing when off.

Design and comparison with AsmJit: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Third-party notice: AsmJit (Zlib) is used only for tests and is not distributed, see [third_party/THIRD_PARTY.txt](third_party/THIRD_PARTY.txt).

Contributing, formatting rules: [CONTRIBUTING.md](CONTRIBUTING.md).
Roadmap (architectures, SIMD, Builder, Compiler, register allocator): [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start

```zig
const std = @import("std");
const pjit = @import("pyrlyn_jit");

var cb = pjit.CodeBuffer.init(gpa);
defer cb.deinit();
const a = pjit.x86_64.Assembler.init(&cb);

// fn add(a: i64, b: i64) i64   (SysV: rdi, rsi -> rax)
try a.movRR(.rax, .rdi);
try a.addRR(.rax, .rsi);
try a.ret();

try cb.finalize();                       // resolve forward-label fixups
const f = try pjit.JitRuntime.add(&cb);    // copy into executable memory
defer f.release();
const add = f.ptr(*const fn (i64, i64) callconv(.c) i64);
std.debug.print("{d}\n", .{add(40, 2)});
```


## Building

Requires **Zig 0.17.0** (pinned in `mise.toml`). `zig build` builds the library, `zig build test` runs the tests.
Full build options, install, pkg-config, CMake and CPM: [docs/BUILDING.md](docs/BUILDING.md).

## Documentation

- [docs/API.md](docs/API.md): core, x86-64, AArch64 and C/C++/Rust API.
- [docs/c-api.md](docs/c-api.md): C ABI reference.
- [docs/SAFETY.md](docs/SAFETY.md): validation, hooks, enabling and disabling.
- [docs/BUILDING.md](docs/BUILDING.md): building and using from C/C++.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): benchmark vs AsmJit.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md): current limitations and implemented features.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md).
