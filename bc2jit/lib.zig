//! Public KXBC Zig API: validate bytes, compile, optionally map executable memory.
const std = @import("std");
pub const bytecode = @import("bytecode.zig");
pub const lower = @import("lower.zig");
pub const driver = @import("driver.zig");
/// KXBC -> IR (`toIr`), used by the optimizer and benchmarks.
pub const lower_ir = @import("lower_ir.zig");
pub const cli = @import("cli.zig");
pub const c_api = @import("c_api.zig");
pub const enabled = @import("pyrlyn_jit").build_options.bc2jit;
pub const Target = lower.Target;
pub const Compiled = driver.Compiled;
pub const Fn = driver.Fn;
pub const Options = driver.Options;
pub const Diagnostic = bytecode.Diagnostic;
pub const Error = bytecode.Error || driver.Error || error{FeatureDisabled};

/// Validates a complete KXBC module. Input bytes may be freed immediately after return.
/// `diag` describes decoder failures; it is reset for every call.
/// The caller owns the result and must call `deinit` exactly once, after all calls finish.
pub fn compile(gpa: std.mem.Allocator, target: Target, bytes: []const u8, options: Options, diag: *Diagnostic) Error!Compiled {
    diag.* = .{};
    if (!enabled) return error.FeatureDisabled;
    const f = try bytecode.decode(gpa, bytes, diag);
    defer f.deinit(gpa);
    diag.* = .{};
    return driver.compileWithOptions(gpa, target, f, options);
}

pub fn errorMessage(e: Error) []const u8 {
    if (e == error.FeatureDisabled) return "bc2jit disabled (rebuild with -Dbc2jit=true)";
    inline for (comptime std.meta.fieldNames(bytecode.Error)) |name| {
        if (e == @field(bytecode.Error, name)) return bytecode.errorMessage(@field(bytecode.Error, name));
    }
    return driver.errorMessage(@errorCast(e));
}
