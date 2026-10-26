//! Build for x86_64-macos and run under Rosetta:
//!   zig build -Dtarget=x86_64-macos run-example_x86_64
const std = @import("std");
const zj = @import("zigjit");
const x = zj.x86_64;

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    var cb = zj.CodeBuffer.init(gpa.allocator());
    defer cb.deinit();
    const a = x.Assembler.init(&cb);

    // fn() i64 { return 42; }
    try a.movRI(.rax, 42);
    try a.ret();
    try show("const42", &cb, fn () callconv(.c) i64, .{});

    // fn(a, b) i64 { return a + b; }   SysV: rdi, rsi
    cb.reset();
    try a.movRR(.rax, .rdi);
    try a.addRR(.rax, .rsi);
    try a.ret();
    try show("add(40, 2)", &cb, fn (i64, i64) callconv(.c) i64, .{ 40, 2 });

    // fn(p: [*]const i64, n) i64 { sum p[0..n] } — labels, jcc, SIB memory operands
    cb.reset();
    const loop = try a.newLabel();
    const done = try a.newLabel();
    try a.xorRR(.rax, .rax);
    try a.xorRR(.rcx, .rcx);
    try a.bind(loop);
    try a.cmpRR(.rcx, .rsi);
    try a.jcc(.ge, done);
    try a.aluRM(.add, .rax, x.ptrIdx(.rdi, .rcx, .x8, 0));
    try a.addRI(.rcx, 1);
    try a.jmp(loop);
    try a.bind(done);
    try a.ret();
    const arr = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try show("sum([1..10])", &cb, fn ([*]const i64, i64) callconv(.c) i64, .{ &arr, arr.len });

    // fn(n) i64 { return fact(n) } — recursive call to a label, push/pop, imul
    cb.reset();
    const fact = try a.newLabel();
    const base = try a.newLabel();
    try a.bind(fact);
    try a.cmpRI(.rdi, 1);
    try a.jcc(.le, base);
    try a.push(.rdi);
    try a.subRI(.rdi, 1);
    try a.call(fact);
    try a.pop(.rdi);
    try a.imulRR(.rax, .rdi);
    try a.ret();
    try a.bind(base);
    try a.movRI(.rax, 1);
    try a.ret();
    try show("fact(10)", &cb, fn (i64) callconv(.c) i64, .{10});
}

fn show(name: []const u8, cb: *zj.CodeBuffer, comptime F: type, args: anytype) !void {
    try cb.finalize();
    const f = try zj.JitRuntime.add(cb);
    defer f.release();
    const r = @call(.auto, f.ptr(*const F), args);
    std.debug.print("{s:<14} = {d} bytes: {x}\n", .{ name, r, cb.code() });
}
