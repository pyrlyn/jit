# Contributing

- Every change goes through a pull request; CI runs Linux first, then Windows and macOS.
- Toolchain: Zig 0.17.0 from `mise.toml`.
- Tests: `zig build test` (add `-Dasmjit-tests=true` for the AsmJit-ported tests).

## Formatting

`zig build fmt` formats and `zig build fmt-check` checks (CI runs it on Linux) `build.zig`,
`src/`, `examples/` and `bench/`. `zig fmt` has no config file, so the exclusions are
`--exclude` paths in `build.zig`:

- `third_party/` (local AsmJit clone, see [third_party/THIRD_PARTY.txt](third_party/THIRD_PARTY.txt)) and `bench/pre-safety/` (frozen snapshot)
- AsmJit-ported tests: `src/asmjit_{x64,x86,a64}_tests.zig`
- Tables generated from AsmJit's instruction database: `src/x86_64/{gp_table,vec_table,isa}.zig`, `src/aarch64/{a64_table,isa}.zig`
- Tests generated from LLVM: `src/{gp,simd,enc,a64_sve,a64_sve_extra}_tests.zig`

Generated files are never edited or reformatted by hand; change the generator in `scripts/` instead.
