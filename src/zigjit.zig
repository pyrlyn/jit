//! ZigJIT — a machine code generator in the spirit of AsmJit.
//! Each target is a separate module compiled in only if enabled via `-Dtargets=`.
const opts = @import("zigjit_options");

pub const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
pub const Label = @import("core/code_buffer.zig").Label;
pub const Arch = @import("core/arch.zig").Arch;
pub const Environment = @import("core/arch.zig").Environment;
/// Safety layer: `Options`, hook helpers (`CountingHooks`, `LogHooks`, `CallbackHooks`).
pub const safety = @import("core/safety.zig");
/// Build-time options (`validate`, `hooks`, enabled backends).
pub const build_options = opts;
pub const JitRuntime = @import("core/jit_runtime.zig");

/// `zigjit.x86_64.Assembler`, `.Gp`, `.ptr`, ... (void if disabled at build time)
pub const x86_64 = if (opts.x86_64) @import("x86_64/assembler.zig") else void;
/// `zigjit.aarch64.Assembler`, `.X`, ... (void if disabled at build time)
pub const aarch64 = if (opts.aarch64) @import("aarch64/assembler.zig") else void;

test {
    _ = @import("core/code_buffer.zig");
    _ = @import("tests.zig");
    _ = @import("enc_tests.zig");
    _ = @import("gp_tests.zig");
    if (opts.x86_64) _ = x86_64;
    if (opts.aarch64) _ = aarch64;
}
