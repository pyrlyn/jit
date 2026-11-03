//! bc2jit as a library module (used by bench/bench_test.zig): decoder, lowering, driver.
pub const bytecode = @import("bytecode.zig");
pub const lower = @import("lower.zig");
pub const driver = @import("driver.zig");
