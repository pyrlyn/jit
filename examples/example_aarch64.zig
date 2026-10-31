//! Native on Apple Silicon:  zig build run-example_aarch64
const std = @import("std");
const pjit = @import("pyrlyn_jit");
const A = pjit.aarch64;

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    var cb = pjit.CodeBuffer.init(gpa.allocator());
    defer cb.deinit();
    const a = A.Assembler.init(&cb);

    try a.movz(.x0, 42, 0);
    try a.ret();
    try show("const42", &cb, fn () callconv(.c) i64, .{});

    cb.reset();
    try a.add(.x0, .x0, .x1);
    try a.ret();
    try show("add(40, 2)", &cb, fn (i64, i64) callconv(.c) i64, .{ 40, 2 });

    cb.reset();
    try a.movImm(.x0, 0x0123_4567_89AB_CDEF);
    try a.ret();
    try show("imm64", &cb, fn () callconv(.c) i64, .{});

    // sum p[0..n]: x0=p, x1=n
    cb.reset();
    const loop = try a.newLabel();
    const done = try a.newLabel();
    try a.movz(.x2, 0, 0);
    try a.bind(loop);
    try a.cbz(.x1, done);
    try a.ldr(.x3, .x0, 0);
    try a.add(.x2, .x2, .x3);
    try a.addImm(.x0, .x0, 8);
    try a.subImm(.x1, .x1, 1);
    try a.b(loop);
    try a.bind(done);
    try a.mov(.x0, .x2);
    try a.ret();
    const arr = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try show("sum([1..10])", &cb, fn ([*]const i64, i64) callconv(.c) i64, .{ &arr, arr.len });
}

fn show(name: []const u8, cb: *pjit.CodeBuffer, comptime F: type, args: anytype) !void {
    try cb.finalize();
    const f = try pjit.JitRuntime.add(cb);
    defer f.release();
    const r = @call(.auto, f.ptr(*const F), args);
    std.debug.print("{s:<14} = {d} bytes: {x}\n", .{ name, r, cb.code() });
}
