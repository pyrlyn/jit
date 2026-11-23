//! Optional LLVM execution backend. No LLVM headers/library are needed unless enabled.
const std = @import("std");
const ir = @import("ir.zig");
pub const enabled = @import("pyrlyn_jit_options").llvm;
pub const Mode = enum { jit, interpreter };
pub const Options = struct {
    mode: Mode = .jit,
    /// Scalar SSA promotion, constant propagation, dead code and CFG simplification.
    optimize: bool = true,
};
pub const Error = error{ OutOfMemory, BackendNotBuilt, UnsupportedIr, InvalidIr, LlvmFailed };

/// Owns a bounded copy of LLVM's diagnostic; no LLVM-owned message escapes the API.
pub const Diagnostic = struct {
    buffer: [2048]u8 = undefined,
    len: usize = 0,
    pub fn message(d: *const Diagnostic) []const u8 {
        return d.buffer[0..d.len];
    }
    pub fn set(d: *Diagnostic, msg: []const u8) void {
        d.len = @min(d.buffer.len, msg.len);
        @memcpy(d.buffer[0..d.len], msg[0..d.len]);
    }
};

pub const Program = if (enabled) @import("llvm_enabled.zig").Program else struct {
    pub fn deinit(_: *@This()) void {}
    pub fn call(_: *const @This(), _: [4]i64) i64 {
        unreachable; // prepare cannot succeed in a build without LLVM
    }
    pub fn irText(_: *const @This(), _: std.mem.Allocator) Error![]u8 {
        return error.BackendNotBuilt;
    }
};

/// Accepts the integer, mutable-register subset emitted by bc2jit's lowering (<=4 args).
/// The returned program owns all LLVM state; the input IR may be freed immediately.
/// LLVM Interpreter has no instruction budget; call only terminating/trusted programs.
pub fn prepare(gpa: std.mem.Allocator, f: *const ir.Func, options: Options, diag: *Diagnostic) Error!Program {
    diag.len = 0;
    if (comptime enabled) return @import("llvm_enabled.zig").prepare(gpa, f, options, diag);
    diag.set("LLVM backend not built; rebuild with -Dllvm-prefix=PREFIX");
    return error.BackendNotBuilt;
}
