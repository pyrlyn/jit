const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const driver = @import("driver.zig");
const cli = @import("cli.zig");
const lower = @import("lower.zig");
const t = std.testing;

test "LLVM: optional backend reports an actionable error" {
    if (comptime jit.llvm.enabled) return error.SkipZigTest;
    var f = jit.ir.Func.init(t.allocator);
    defer f.deinit();
    var diag: jit.llvm.Diagnostic = .{};
    try t.expectError(error.BackendNotBuilt, jit.llvm.prepare(t.allocator, &f, .{}, &diag));
    try t.expect(std.mem.indexOf(u8, diag.message(), "-Dllvm-prefix") != null);
}

test "LLVM: v0 constant folding is observable and both engines agree with eval" {
    if (comptime !jit.llvm.enabled) return error.SkipZigTest;
    const insts: []const bc.Inst = &.{ .{ .push_i64 = 6 }, .{ .push_i64 = 7 }, .mul, .ret };
    const bytes = try bc.encode(t.allocator, .{ .arg_count = 0, .insts = insts });
    defer t.allocator.free(bytes);
    var diag: bc.Diagnostic = .{};
    const f = try bc.decode(t.allocator, bytes, &diag);
    defer f.deinit(t.allocator);
    for ([_]jit.llvm.Mode{ .jit, .interpreter }) |mode| {
        for ([_]bool{ false, true }) |optimize| {
            var llvm_diag: jit.llvm.Diagnostic = .{};
            var p = try driver.compileLlvm(t.allocator, f, .{ .mode = mode, .optimize = optimize }, &llvm_diag);
            defer p.deinit();
            const text = try p.irText(t.allocator);
            defer t.allocator.free(text);
            if (optimize) {
                try t.expect(std.mem.indexOf(u8, text, "ret i64 42") != null);
                try t.expect(std.mem.indexOf(u8, text, "alloca") == null);
                try t.expect(std.mem.indexOf(u8, text, "mul i64") == null);
            } else {
                try t.expect(std.mem.indexOf(u8, text, "alloca") != null);
                try t.expect(std.mem.indexOf(u8, text, "mul i64") != null);
            }
            // Reuse one engine, including unused extra arguments.
            try t.expectEqual(try bc.eval(f, .{ 1, 2, 3, 4 }, 100), p.call(.{ 1, 2, 3, 4 }));
            try t.expectEqual(@as(i64, 42), p.call(.{ 0, 0, 0, 0 }));
        }
    }
}

test "LLVM: malformed or unsupported IR is rejected before execution" {
    if (comptime !jit.llvm.enabled) return error.SkipZigTest;
    var f = jit.ir.Func.init(t.allocator);
    defer f.deinit();
    var diag: jit.llvm.Diagnostic = .{};
    try t.expectError(error.InvalidIr, jit.llvm.prepare(t.allocator, &f, .{}, &diag));
    const block = try f.addBlock("entry");
    try f.emit(block, .{ .fconst = .{ .dst = 0, .ty = .f64, .bits = 0 } });
    try f.emit(block, .{ .ret = 0 });
    try t.expectError(error.UnsupportedIr, jit.llvm.prepare(t.allocator, &f, .{}, &diag));
}

fn allocationFailure(gpa: std.mem.Allocator) !void {
    var f = try jit.ir.parse(t.allocator, "func f 1\nentry:\n ret v0\n");
    defer f.deinit();
    var diag: jit.llvm.Diagnostic = .{};
    var p = try jit.llvm.prepare(gpa, &f, .{}, &diag);
    defer p.deinit();
    try t.expectEqual(@as(i64, -7), p.call(.{ -7, 0, 0, 0 }));
}

test "LLVM: allocation failures release temporary Zig and LLVM state" {
    if (comptime !jit.llvm.enabled) return error.SkipZigTest;
    try t.checkAllAllocationFailures(t.allocator, allocationFailure, .{});
}

test "cli: LLVM backends, optimization and incompatible flags" {
    const host = lower.Target.host() orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for ([_][]const u8{ "llvm-jit", "llvm-interpreter" }) |backend| {
        const parsed = try cli.parse(arena.allocator(), &.{ "--arch", host.flagName(), "--backend", backend, "--no-opt", "f.kxbc", "-7" });
        try t.expect(parsed == .run);
        try t.expect(parsed.run.backend != .pyrlyn);
        try t.expect(!parsed.run.optimize);
        try t.expectEqual(@as(i64, -7), parsed.run.args[0]);
    }
    const bad = [_][]const []const u8{
        &.{ "-a", host.flagName(), "--backend" },
        &.{ "-a", host.flagName(), "--backend=unknown", "f.kxbc" },
        &.{ "-a", host.flagName(), "--no-opt", "f.kxbc" },
        &.{ "-a", host.flagName(), "--backend=llvm-jit", "--no-jit", "f.kxbc" },
        &.{ "-a", host.flagName(), "--backend=llvm-interpreter", "--no-jit", "f.kxbc" },
        &.{ "-a", host.flagName(), "--backend=llvm-jit", "-o", "out.bin", "f.kxbc" },
        &.{ "-a", if (host == .riscv64) "aarch64" else "riscv", "--backend=llvm-interpreter", "f.kxbc" },
    };
    for (bad) |args| try t.expect((try cli.parse(arena.allocator(), args)) == .usage_error);
}
