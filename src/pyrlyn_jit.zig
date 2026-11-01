//! pyrlyn-jit — a machine code generator in the spirit of AsmJit.
//! Each target is a separate module compiled in only if enabled via `-Dtargets=`.
const opts = @import("pyrlyn_jit_options");

pub const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
pub const Label = @import("core/code_buffer.zig").Label;
pub const Arch = @import("core/arch.zig").Arch;
pub const Environment = @import("core/arch.zig").Environment;
/// Safety layer: `Options`, hook helpers (`CountingHooks`, `LogHooks`, `CallbackHooks`).
pub const safety = @import("core/safety.zig");
/// Build-time options (`validate`, `hooks`, enabled backends).
pub const build_options = opts;
/// Arch-agnostic linear-scan register allocator (vregs -> physical registers / spill slots).
pub const regalloc = @import("core/regalloc.zig");
/// Compiler IR: text parser, verifier, printer, reference interpreter.
pub const ir = @import("core/ir.zig");
pub const JitRuntime = @import("core/jit_runtime.zig");

/// `pyrlyn_jit.x86_64.Assembler`, `.Gp`, `.ptr`, ... (void if disabled at build time)
pub const x86_64 = if (opts.x86_64) @import("x86_64/assembler.zig") else void;
/// `pyrlyn_jit.x86_64_builder.Builder`: node-list layer over the x86-64 Assembler (void if disabled).
pub const x86_64_builder = if (opts.x86_64) @import("x86_64/builder.zig") else void;
/// `pyrlyn_jit.x86_64_compiler.compile`: IR -> regalloc -> Builder -> machine code (void if disabled).
pub const x86_64_compiler = if (opts.x86_64) @import("x86_64/compiler.zig") else void;
/// `pyrlyn_jit.x86.Assembler`: 32-bit x86 (part of the x86_64 backend's encoder; void if disabled).
pub const x86 = if (opts.x86_64) @import("x86/assembler.zig") else void;
/// `pyrlyn_jit.riscv.Assembler`: RV64IM (always compiled; no host dependency).
pub const riscv = @import("riscv/assembler.zig");
/// `pyrlyn_jit.aarch64.Assembler`, `.X`, ... (void if disabled at build time)
pub const aarch64 = if (opts.aarch64) @import("aarch64/assembler.zig") else void;

test {
    _ = @import("core/code_buffer.zig");
    _ = @import("core/regalloc.zig");
    _ = @import("core/ir.zig");
    _ = @import("tests.zig");
    _ = @import("enc_tests.zig");
    _ = @import("gp_tests.zig");
    _ = @import("simd_tests.zig");
    _ = @import("vec_table_tests.zig");
    _ = @import("a64_sve_tests.zig");
    _ = @import("a64_sve_extra_tests.zig");
    _ = @import("a64_table_tests.zig");
    if (opts.asmjit_tests) _ = @import("asmjit_x64_tests.zig");
    if (opts.asmjit_tests) _ = @import("asmjit_x86_tests.zig");
    if (opts.asmjit_tests) _ = @import("asmjit_a64_tests.zig");
    if (opts.gp_fuzz) _ = @import("gp_fuzz.zig");
    _ = @import("fuzz.zig");
    if (opts.x86_64) _ = x86_64;
    if (opts.x86_64) _ = x86_64_builder;
    if (opts.x86_64) _ = x86_64_compiler;
    if (opts.x86_64) _ = x86;
    _ = riscv;
    if (opts.aarch64) _ = aarch64;
}
