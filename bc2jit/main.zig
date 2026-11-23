//! `bc2jit`: compile a KXBC bytecode function for `--arch`; run it when that is the host.
const std = @import("std");
const api = @import("bc2jit");
const cli = api.cli;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    const parsed = try cli.parse(arena, argv[1..]);
    const o = switch (parsed) {
        .help => return std.Io.File.stdout().writeStreamingAll(io, cli.usage),
        .usage_error => |msg| {
            std.debug.print("{s}\n", .{msg});
            std.process.exit(cli.exit_usage);
        },
        .run => |o| o,
    };
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, o.input, arena, .unlimited) catch |e|
        fail("cannot read '{s}': {s}", .{ o.input, @errorName(e) });
    var diag: api.Diagnostic = .{};
    if (o.backend != .pyrlyn) {
        const f = api.bytecode.decode(arena, bytes, &diag) catch |e|
            fail("{s}: offset {d}: {s}", .{ o.input, diag.offset, api.bytecode.errorMessage(e) });
        defer f.deinit(arena);
        var llvm_diag: @import("pyrlyn_jit").llvm.Diagnostic = .{};
        var program = api.driver.compileLlvm(arena, f, .{
            .mode = if (o.backend == .llvm_jit) .jit else .interpreter,
            .optimize = o.optimize,
        }, &llvm_diag) catch |e| fail("LLVM: {s}: {s}", .{ @errorName(e), llvm_diag.message() });
        defer program.deinit();
        try std.Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(arena, "{d}\n", .{program.call(o.args)}));
        return;
    }
    var c = api.compile(arena, o.target, bytes, .{ .executable = o.executable, .opt_level = o.opt_level }, &diag) catch |e|
        fail("{s}: offset {d}: {s}", .{ o.input, diag.offset, api.errorMessage(e) });
    defer c.deinit();
    if (o.output) |path| std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = c.code() }) catch |e|
        fail("cannot write '{s}': {s}", .{ path, @errorName(e) });
    const out = std.Io.File.stdout();
    if (c.exec != null) {
        try out.writeStreamingAll(io, try std.fmt.allocPrint(arena, "{d}\n", .{c.call(o.args)}));
    } else if (o.output == null) {
        try out.writeStreamingAll(io, try std.fmt.allocPrint(arena, "{x}\n", .{c.code()}));
    }
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("bc2jit: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}
