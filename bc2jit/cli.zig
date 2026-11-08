//! Command-line parsing for `bc2jit`, kept free of I/O so it can be unit-tested.
const std = @import("std");
const lower = @import("lower.zig");

/// Usage text printed by `-h`/`--help`.
pub const usage =
    \\usage: bc2jit -a ARCH [--no-jit] [-o FILE] INPUT [ARG...]
    \\
    \\Compile a KXBC (v0 or v1) bytecode function to machine code with pyrlyn-jit.
    \\
    \\When ARCH is the host architecture, the function is executed with up to four
    \\integer ARGs (missing ones are 0) and its i64 result is printed on stdout.
    \\Otherwise the machine code is printed as hex on stdout (or written raw with -o).
    \\
    \\options:
    \\  -a, --arch ARCH     target architecture: x86-64, aarch64, riscv (required)
    \\  -o, --output FILE   write the raw machine code to FILE (also when executing)
    \\  --backend NAME     pyrlyn (default), llvm-jit or llvm-interpreter
    \\  --no-opt           disable LLVM optimization (LLVM backends only)
    \\                     LLVM requires --arch = host; -o is not supported
    \\      --no-jit       emit code without mapping or executing it (pyrlyn only)
    \\  -h, --help          show this help and exit
    \\
    \\exit status: 0 success, 1 compile or I/O error, 2 usage error
    \\
;

/// Parsed command line.
pub const Options = struct {
    pub const Backend = enum { pyrlyn, llvm_jit, llvm_interpreter };
    target: lower.Target,
    backend: Backend = .pyrlyn,
    optimize: bool = true,
    input: []const u8,
    output: ?[]const u8 = null,
    /// Integer arguments for execution (at most `bytecode.max_args`).
    args: [4]i64 = .{ 0, 0, 0, 0 },
    arg_count: u8 = 0,
    executable: bool = true,
};

/// Result of parsing: run with options, show help, or fail with a message (exit status 2).
pub const Parsed = union(enum) {
    run: Options,
    help,
    /// Usage error; `msg` is the full line to print on stderr (without trailing newline).
    usage_error: []const u8,
};

/// Exit status for usage errors.
pub const exit_usage = 2;

/// Parses `argv` (without the program name). Messages are allocated from `arena`.
pub fn parse(arena: std.mem.Allocator, argv: []const []const u8) error{OutOfMemory}!Parsed {
    var target: ?lower.Target = null;
    var output: ?[]const u8 = null;
    var executable = true;
    var backend: Options.Backend = .pyrlyn;
    var optimize = true;
    var positional: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    var only_positional = false;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (only_positional or a.len < 2 or a[0] != '-' or isNumber(a)) {
            try positional.append(arena, a);
            continue;
        }
        if (std.mem.eql(u8, a, "--")) {
            only_positional = true;
            continue;
        }
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) return .help;
        if (std.mem.eql(u8, a, "--no-opt")) {
            optimize = false;
            continue;
        }
        const backend_val = optionValue(argv, &i, a, "--backend", "--backend") catch
            return .{ .usage_error = "bc2jit: option --backend requires a value (pyrlyn, llvm-jit, llvm-interpreter)" };
        if (backend_val) |v| {
            backend = if (std.mem.eql(u8, v, "pyrlyn")) .pyrlyn else if (std.mem.eql(u8, v, "llvm-jit")) .llvm_jit else if (std.mem.eql(u8, v, "llvm-interpreter")) .llvm_interpreter else return .{ .usage_error = "bc2jit: unknown --backend (allowed: pyrlyn, llvm-jit, llvm-interpreter)" };
            continue;
        }
        if (std.mem.eql(u8, a, "--no-jit")) {
            executable = false;
            continue;
        }
        const arch_val = optionValue(argv, &i, a, "-a", "--arch") catch
            return .{ .usage_error = "bc2jit: option --arch requires a value (allowed: " ++ lower.allowed_flags ++ ")" };
        if (arch_val) |v| {
            target = lower.Target.fromFlag(v) orelse return .{ .usage_error = try std.fmt.allocPrint(
                arena,
                "bc2jit: unknown --arch value '{s}' (allowed: " ++ lower.allowed_flags ++ ")",
                .{v},
            ) };
            continue;
        }
        const out_val = optionValue(argv, &i, a, "-o", "--output") catch
            return .{ .usage_error = "bc2jit: option --output requires a value" };
        if (out_val) |v| {
            output = v;
            continue;
        }
        return .{ .usage_error = try std.fmt.allocPrint(arena, "bc2jit: unknown option '{s}' (see --help)", .{a}) };
    }
    const t = target orelse return .{ .usage_error = "bc2jit: missing required option --arch (allowed: " ++ lower.allowed_flags ++ ")" };
    if (backend != .pyrlyn and !executable) return .{ .usage_error = "bc2jit: --no-jit requires the pyrlyn backend" };
    if (backend == .pyrlyn and !optimize) return .{ .usage_error = "bc2jit: --no-opt requires an LLVM backend" };
    if (backend != .pyrlyn and output != null) return .{ .usage_error = "bc2jit: -o is not supported by LLVM backends" };
    if (backend != .pyrlyn and lower.Target.host() != t) return .{ .usage_error = "bc2jit: LLVM backends require --arch to match the host" };
    if (positional.items.len == 0) return .{ .usage_error = "bc2jit: missing INPUT file (see --help)" };
    const rest = positional.items[1..];
    if (rest.len > 4) return .{ .usage_error = "bc2jit: at most 4 ARGs are allowed" };
    var o: Options = .{ .target = t, .backend = backend, .optimize = optimize, .input = positional.items[0], .output = output, .arg_count = @intCast(rest.len), .executable = executable };
    for (rest, 0..) |s, k| {
        o.args[k] = std.fmt.parseInt(i64, s, 0) catch return .{ .usage_error = try std.fmt.allocPrint(
            arena,
            "bc2jit: ARG '{s}' is not a 64-bit integer",
            .{s},
        ) };
    }
    return .{ .run = o };
}

fn isNumber(a: []const u8) bool {
    _ = std.fmt.parseInt(i64, a, 0) catch return false;
    return true;
}

/// Matches `-x VALUE`, `--long VALUE`, `--long=VALUE`. Returns null if `a` is not this option,
/// error.MissingValue if it is but no value follows.
fn optionValue(argv: []const []const u8, i: *usize, a: []const u8, short: []const u8, long: []const u8) error{MissingValue}!?[]const u8 {
    if (std.mem.startsWith(u8, a, long) and a.len > long.len and a[long.len] == '=') {
        const v = a[long.len + 1 ..];
        return if (v.len == 0) error.MissingValue else v;
    }
    if (!std.mem.eql(u8, a, short) and !std.mem.eql(u8, a, long)) return null;
    if (i.* + 1 >= argv.len) return error.MissingValue;
    i.* += 1;
    return argv[i.*];
}
