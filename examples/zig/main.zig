//! ZigJIT Zig example (native API): JIT-compiles `add(a, b)` for the host and calls it.
//! zig build examples   (also runs as part of `zig build test`)
const std = @import("std");
const builtin = @import("builtin");
const zj = @import("zigjit");

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    var cb = zj.CodeBuffer.init(gpa.allocator());
    defer cb.deinit();

    switch (builtin.cpu.arch) {
        .aarch64 => {
            const a = zj.aarch64.Assembler.init(&cb);
            const done = try a.newLabel();
            try a.add(.x0, .x0, .x1);
            try a.b(done);
            try cb.emitBytes(&.{ 0, 0, 0, 0 }); // udf #0, skipped
            try a.bind(done);
            try a.ret();
        },
        .x86_64 => {
            const a = zj.x86_64.Assembler.init(&cb);
            try a.movRR(.rax, .rdi);
            try a.addRR(.rax, .rsi);
            try a.ret();
        },
        else => @compileError("unsupported host"),
    }
    // Errors are values: an unbound label is reported, nothing is patched.
    try cb.finalize();

    const f = try zj.JitRuntime.add(&cb); // W^X executable copy
    defer f.release();
    const r = f.ptr(*const fn (i64, i64) callconv(.c) i64)(40, 2);
    std.debug.print("zigjit (Zig): add(40, 2) = {d}\n", .{r});
    if (r != 42) return error.WrongResult;
}
