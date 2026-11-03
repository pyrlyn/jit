//! KXBC v0: a compact stack bytecode for single integer functions, the proposed hand-off format
//! from the KotlinX AoT pipeline to the JIT (see bc2jit/README.md for the byte layout).
//!
//! A module is one function over `i64`: up to `max_args` arguments, an operand stack, and a
//! trailing `ret`. `decode` validates the whole stream (opcodes, operands, stack depth) so the
//! lowering never sees an ill-formed program.
const std = @import("std");

/// File magic, the first four bytes of every module.
pub const magic = "KXBC";
/// The only format version this decoder accepts.
pub const version: u8 = 0;
/// Header size in bytes: magic, version, argument count, little-endian u16 code length.
pub const header_len = 8;
/// Most arguments a function may take (they arrive in the first four ABI argument registers).
pub const max_args = 4;
/// Deepest operand stack a function may use (one physical register per stack slot).
pub const max_depth = 5;

/// Bytecode opcodes. Operands follow the opcode byte in little-endian order.
pub const Opcode = enum(u8) {
    /// Push a 64-bit immediate (operand: i64).
    push_i64 = 0x01,
    /// Push argument `n` (operand: u8, `n < arg_count`).
    push_arg = 0x02,
    /// Pop b, pop a, push a + b (wrapping).
    add = 0x10,
    /// Pop b, pop a, push a - b (wrapping).
    sub = 0x11,
    /// Pop b, pop a, push a * b (wrapping, low 64 bits).
    mul = 0x12,
    /// Push a copy of the top of stack.
    dup = 0x20,
    /// Discard the top of stack.
    drop = 0x21,
    /// Pop the top of stack and return it; must be the last instruction.
    ret = 0xF0,
};

/// One decoded instruction.
pub const Inst = union(Opcode) {
    push_i64: i64,
    push_arg: u8,
    add,
    sub,
    mul,
    dup,
    drop,
    ret,

    /// Operand-stack depth change of this instruction.
    pub fn stackEffect(i: Inst) i8 {
        return switch (i) {
            .push_i64, .push_arg, .dup => 1,
            .add, .sub, .mul, .drop, .ret => -1,
        };
    }
    /// Operands this instruction needs on the stack before it runs.
    pub fn pops(i: Inst) u8 {
        return switch (i) {
            .push_i64, .push_arg => 0,
            .dup, .drop, .ret => 1,
            .add, .sub, .mul => 2,
        };
    }
};

/// A decoded, validated function.
pub const Function = struct {
    arg_count: u8,
    insts: []const Inst,

    /// Frees the instruction slice allocated by `decode`.
    pub fn deinit(f: Function, gpa: std.mem.Allocator) void {
        gpa.free(f.insts);
    }
};

/// Decoding/validation failures, each with a message from `errorMessage`.
pub const Error = error{
    OutOfMemory,
    BadMagic,
    UnsupportedVersion,
    TooManyArgs,
    Truncated,
    TrailingBytes,
    UnknownOpcode,
    ArgOutOfRange,
    StackUnderflow,
    StackTooDeep,
    MissingRet,
    CodeAfterRet,
};

/// Human-readable description of a decode error.
pub fn errorMessage(e: Error) []const u8 {
    return switch (e) {
        error.OutOfMemory => "out of memory",
        error.BadMagic => "not a KXBC module (bad magic)",
        error.UnsupportedVersion => "unsupported KXBC version (expected 0)",
        error.TooManyArgs => "argument count exceeds 4",
        error.Truncated => "module truncated",
        error.TrailingBytes => "bytes after the declared code length",
        error.UnknownOpcode => "unknown opcode",
        error.ArgOutOfRange => "push_arg index out of range",
        error.StackUnderflow => "operand stack underflow",
        error.StackTooDeep => "operand stack deeper than 5",
        error.MissingRet => "function does not end with ret",
        error.CodeAfterRet => "instructions after ret",
    };
}

/// Where decoding stopped: the byte offset of the failing instruction (or header field).
pub const Diagnostic = struct { offset: usize = 0 };

/// Decodes and validates `bytes`. On error, `diag.offset` points at the offending byte.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, diag: *Diagnostic) Error!Function {
    diag.offset = 0;
    if (bytes.len < header_len) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..4], magic)) return error.BadMagic;
    diag.offset = 4;
    if (bytes[4] != version) return error.UnsupportedVersion;
    diag.offset = 5;
    const arg_count = bytes[5];
    if (arg_count > max_args) return error.TooManyArgs;
    diag.offset = 6;
    const code_len = std.mem.readInt(u16, bytes[6..8], .little);
    // Widen before adding: `header_len + code_len` in u16 overflows for code_len > 65527.
    const total = @as(usize, header_len) + code_len;
    if (bytes.len < total) return error.Truncated;
    if (bytes.len > total) return error.TrailingBytes;
    const code = bytes[header_len..];

    var list: std.ArrayList(Inst) = .empty;
    errdefer list.deinit(gpa);
    var pc: usize = 0;
    var depth: u8 = 0;
    var saw_ret = false;
    while (pc < code.len) {
        diag.offset = header_len + pc;
        if (saw_ret) return error.CodeAfterRet;
        const op = std.enums.fromInt(Opcode, code[pc]) orelse return error.UnknownOpcode;
        pc += 1;
        const inst: Inst = switch (op) {
            .push_i64 => blk: {
                if (code.len - pc < 8) return error.Truncated;
                const v = std.mem.readInt(i64, code[pc..][0..8], .little);
                pc += 8;
                break :blk .{ .push_i64 = v };
            },
            .push_arg => blk: {
                if (code.len - pc < 1) return error.Truncated;
                const n = code[pc];
                pc += 1;
                if (n >= arg_count) return error.ArgOutOfRange;
                break :blk .{ .push_arg = n };
            },
            .add => .add,
            .sub => .sub,
            .mul => .mul,
            .dup => .dup,
            .drop => .drop,
            .ret => .ret,
        };
        if (depth < inst.pops()) return error.StackUnderflow;
        const next: i16 = @as(i16, depth) + inst.stackEffect();
        if (next > max_depth) return error.StackTooDeep;
        depth = @intCast(next);
        if (inst == .ret) saw_ret = true;
        try list.append(gpa, inst);
    }
    diag.offset = header_len + pc;
    if (!saw_ret) return error.MissingRet;
    return .{ .arg_count = arg_count, .insts = try list.toOwnedSlice(gpa) };
}

/// Encodes `f` into a fresh module (header + code). Inverse of `decode`.
pub fn encode(gpa: std.mem.Allocator, f: Function) error{ OutOfMemory, CodeTooLarge }![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, magic);
    try out.appendSlice(gpa, &.{ version, f.arg_count, 0, 0 });
    for (f.insts) |i| {
        try out.append(gpa, @backingInt(std.meta.activeTag(i)));
        switch (i) {
            .push_i64 => |v| {
                var b: [8]u8 = undefined;
                std.mem.writeInt(i64, &b, v, .little);
                try out.appendSlice(gpa, &b);
            },
            .push_arg => |n| try out.append(gpa, n),
            else => {},
        }
    }
    const code_len = out.items.len - header_len;
    if (code_len > std.math.maxInt(u16)) return error.CodeTooLarge;
    std.mem.writeInt(u16, out.items[6..8], @intCast(code_len), .little);
    return out.toOwnedSlice(gpa);
}
