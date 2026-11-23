//! C ABI for the public Zig API; see include/bc2jit.h.
const std = @import("std");
const api = @import("lib.zig");
const gpa = std.heap.c_allocator;
const Handle = if (api.enabled) struct { compiled: api.Compiled } else opaque {};
const Diagnostic = extern struct { offset: usize = 0, message: ?[*:0]const u8 = null };
const Status = enum(c_int) { ok, invalid_argument, disabled, invalid_bytecode, backend_not_built, out_of_memory, compile_failed, map_failed, unsupported_host };
fn value(s: Status) c_int {
    return @backingInt(s);
}

export fn pjit_bc2jit_enabled() c_int {
    return @intFromBool(api.enabled);
}

export fn pjit_bc2jit_compile(bytes: ?[*]const u8, len: usize, target: c_int, flags: u32, out: ?*?*Handle, diagnostic: ?*Diagnostic) c_int {
    if (diagnostic) |d| d.* = .{};
    const o = out orelse return value(.invalid_argument);
    o.* = null;
    if (!api.enabled) return value(.disabled);
    if (flags & ~@as(u32, 1) != 0) return value(.invalid_argument);
    const t: api.Target = switch (target) {
        0 => api.Target.host() orelse return value(.unsupported_host),
        1 => .x86_64,
        2 => .aarch64,
        3 => .riscv64,
        else => return value(.invalid_argument),
    };
    const input = if (bytes) |b| b[0..len] else if (len == 0) &.{} else return value(.invalid_argument);
    var diag: api.Diagnostic = .{};
    var compiled = api.compile(gpa, t, input, .{ .executable = flags & 1 == 0 }, &diag) catch |e| {
        if (diagnostic) |d| d.* = .{ .offset = diag.offset, .message = @errorName(e) };
        return value(switch (e) {
            error.FeatureDisabled => .disabled,
            error.BackendNotBuilt => .backend_not_built,
            error.OutOfMemory => .out_of_memory,
            error.MapFailed, error.ProtectFailed => .map_failed,
            else => blk: {
                inline for (comptime std.meta.fieldNames(api.bytecode.Error)) |name| {
                    if (e == @field(api.bytecode.Error, name)) break :blk .invalid_bytecode;
                }
                break :blk .compile_failed;
            },
        });
    };
    const h = gpa.create(Handle) catch {
        compiled.deinit();
        return value(.out_of_memory);
    };
    h.* = .{ .compiled = compiled };
    o.* = h;
    return value(.ok);
}

export fn pjit_bc2jit_code(handle: ?*const Handle, len: ?*usize) ?[*]const u8 {
    if (len) |n| n.* = 0;
    if (!api.enabled) return null;
    const h = handle orelse return null;
    const code = h.compiled.code();
    if (len) |n| n.* = code.len;
    return code.ptr;
}

export fn pjit_bc2jit_entry(handle: ?*const Handle) ?api.Fn {
    if (!api.enabled) return null;
    const h = handle orelse return null;
    return h.compiled.entry();
}

export fn pjit_bc2jit_free(handle: ?*Handle) void {
    if (!api.enabled) return;
    const h = handle orelse return;
    h.compiled.deinit();
    gpa.destroy(h);
}
