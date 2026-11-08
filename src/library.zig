//! Export both C APIs from the same core module, sharing its types and implementation.
comptime {
    _ = @import("pyrlyn_jit").c_api;
    _ = @import("bc2jit").c_api;
}
