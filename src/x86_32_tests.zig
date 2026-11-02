//! Test root for a 32-bit x86 process (`zig build test-x86-32`, `-target x86-linux`): runs the
//! x86 Compiler's execution tests natively in 32-bit mode (skipped by the 64-bit suite).
test {
    _ = @import("x86/compiler.zig");
}
