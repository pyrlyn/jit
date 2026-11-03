//! Bytecode -> machine code -> (host only) executable function.
const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const lower = @import("lower.zig");

/// Signature of every compiled function: four i64 arguments (unused ones ignored), i64 result.
pub const Fn = *const fn (i64, i64, i64, i64) callconv(.c) i64;

/// Machine code for one function, plus a callable mapping when the target can run here.
pub const Compiled = struct {
    cb: jit.CodeBuffer,
    /// Present only when `target.canExecute()`; never created for a foreign target.
    exec: ?jit.JitRuntime.Function,

    /// Finalized machine code bytes.
    pub fn code(c: *const Compiled) []const u8 {
        return c.cb.code();
    }
    /// Calls the function; only valid when `exec != null`.
    pub fn call(c: *const Compiled, args: [4]i64) i64 {
        const f = c.exec.?.ptr(Fn);
        return f(args[0], args[1], args[2], args[3]);
    }
    /// Releases the executable mapping and the code buffer.
    pub fn deinit(c: *Compiled) void {
        if (c.exec) |e| e.release();
        c.cb.deinit();
    }
};

/// Errors from `compile`.
pub const Error = lower.Error || jit.JitRuntime.Error;

/// Lowers `f` for `target`; maps it executable only if the target is the host.
pub fn compile(gpa: std.mem.Allocator, target: lower.Target, f: bc.Function) Error!Compiled {
    var cb = jit.CodeBuffer.init(gpa);
    errdefer cb.deinit();
    try lower.lower(target, f, &cb);
    const exec: ?jit.JitRuntime.Function = if (target.canExecute()) try jit.JitRuntime.add(&cb) else null;
    return .{ .cb = cb, .exec = exec };
}

/// Message for a `compile` error.
pub fn errorMessage(e: Error) []const u8 {
    return switch (e) {
        error.BackendNotBuilt => "backend not compiled into pyrlyn-jit (rebuild with -Dtargets=x86_64,aarch64)",
        error.MapFailed, error.ProtectFailed => "could not map executable memory",
        error.OutOfMemory => "out of memory",
        else => "machine code emission failed",
    };
}
