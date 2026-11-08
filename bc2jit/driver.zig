//! Bytecode -> machine code -> (host only) executable function.
const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const lower = @import("lower.zig");
const lower_ir = @import("lower_ir.zig");

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
    /// Callable C-ABI entry point, or null for code-only/foreign targets.
    pub fn entry(c: *const Compiled) ?Fn {
        return if (c.exec) |e| e.ptr(Fn) else null;
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
pub const Error = lower.Error || lower_ir.Error || jit.JitRuntime.Error;

/// Lowers `f` for `target`; maps it executable only if the target is the host.
/// v0 modules use the direct register-per-slot lowering (`lower.zig`, SysV on x86-64); v1
/// modules go through the IR Compilers (`lower_ir.zig`, host C ABI incl. Win64).
pub fn compile(gpa: std.mem.Allocator, target: lower.Target, f: bc.Function) Error!Compiled {
    return compileWithOptions(gpa, target, f, .{});
}

pub const Options = struct {
    /// False emits code without allocating executable memory, even on the host.
    executable: bool = true,
};

pub fn compileWithOptions(gpa: std.mem.Allocator, target: lower.Target, f: bc.Function, options: Options) Error!Compiled {
    var cb = jit.CodeBuffer.init(gpa);
    errdefer cb.deinit();
    if (f.version == bc.version) try lower.lower(target, f, &cb) else try lower_ir.lowerIr(gpa, target, f, &cb);
    const runnable = if (f.version == bc.version) target.canExecute() else lower.Target.host() == target;
    const exec: ?jit.JitRuntime.Function = if (options.executable and runnable) try jit.JitRuntime.add(&cb) else null;
    return .{ .cb = cb, .exec = exec };
}

/// Optional LLVM backend. Optimizes once, then reuses the program for JIT/interpreter calls.
/// `f` must have been validated by bytecode.decode. Always uses the host target.
pub fn compileLlvm(gpa: std.mem.Allocator, f: bc.Function, options: jit.llvm.Options, diag: *jit.llvm.Diagnostic) (lower_ir.Error || jit.llvm.Error)!jit.llvm.Program {
    diag.len = 0;
    var g = try lower_ir.toIr(gpa, f);
    defer g.deinit();
    return jit.llvm.prepare(gpa, &g, options, diag);
}

/// Message for a `compile` error.
pub fn errorMessage(e: Error) []const u8 {
    return switch (e) {
        error.BackendNotBuilt => "backend not compiled into pyrlyn-jit (rebuild with -Dtargets=x86_64,aarch64)",
        error.MapFailed, error.ProtectFailed => "could not map executable memory",
        error.OutOfMemory => "out of memory",
        error.CompileFailed => "IR compiler rejected the function",
        else => "machine code emission failed",
    };
}
