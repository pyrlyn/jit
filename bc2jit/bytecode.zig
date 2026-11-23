//! KXBC: a compact stack bytecode for single integer functions, the proposed hand-off format
//! from the KotlinX AoT pipeline to the JIT (see bc2jit/README.md for the byte layout).
//!
//! A module is one function over `i64`: up to `max_args` arguments and an operand stack.
//! - v0: straight-line code (push, add/sub/mul, dup/drop) and a trailing `ret`; depth <= 5.
//! - v1: v0 plus division, bitwise ops and shifts, `swap`, 16 locals, `jmp`/`br_if` and `ret`
//!   anywhere; depth <= 64. Every instruction must be reachable and every join must see the same
//!   stack depth (checked by a data-flow pass).
//! `decode` validates the whole stream so the lowering never sees an ill-formed program.
//! `eval` is the reference interpreter used by the differential tests.
const std = @import("std");

/// File magic, the first four bytes of every module.
pub const magic = "KXBC";
/// Format version of v0 modules (the straight-line subset).
pub const version: u8 = 0;
/// Format version with control flow, locals and the full integer op set.
pub const version_v1: u8 = 1;
/// Newest version this decoder accepts.
pub const max_version: u8 = version_v1;
/// Header size in bytes: magic, version, argument count, little-endian u16 code length.
pub const header_len = 8;
/// Most arguments a function may take (they arrive in the first four ABI argument registers).
pub const max_args = 4;
/// Deepest operand stack a v0 function may use (one physical register per stack slot).
pub const max_depth = 5;
/// Deepest operand stack a v1 function may use (lowered through the IR register allocator).
pub const max_depth_v1 = 64;
/// Locals of a v1 function (`load_local`/`store_local` index; all start at 0).
pub const max_locals = 16;

/// `br_if` condition (signed `lt..ge`, unsigned `ult..uge`), comparing `a` (below) with `b` (top).
pub const Cond = enum(u8) { eq, ne, lt, le, gt, ge, ult, ule, ugt, uge };

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
    /// v1. Pop b, pop a, push a / b truncated; `a / 0 = 0`, `minInt / -1 = minInt`.
    div = 0x13,
    /// v1. Pop b, pop a, push a % b (sign of a); `a % 0 = a`, `a % -1 = 0`.
    rem = 0x14,
    /// v1. Bitwise and / or / xor.
    @"and" = 0x15,
    @"or" = 0x16,
    xor = 0x17,
    /// v1. Shifts; the count is taken mod 64 (`shr` logical, `sar` arithmetic).
    shl = 0x18,
    shr = 0x19,
    sar = 0x1A,
    /// Push a copy of the top of stack.
    dup = 0x20,
    /// Discard the top of stack.
    drop = 0x21,
    /// v1. Exchange the two top values.
    swap = 0x22,
    /// v1. Push local `n` (operand: u8, `n < max_locals`).
    load_local = 0x30,
    /// v1. Pop into local `n` (operand: u8).
    store_local = 0x31,
    /// v1. Jump to code offset `t` (operand: u16 LE, an instruction start).
    jmp = 0x40,
    /// v1. Pop b, pop a; jump to `t` if `cond(a, b)` (operands: u8 cond, u16 LE target).
    br_if = 0x41,
    /// Pop the top of stack and return it (v0: must be the last instruction).
    ret = 0xF0,

    /// Opcodes a v0 module may use.
    pub fn inV0(o: Opcode) bool {
        return switch (o) {
            .push_i64, .push_arg, .add, .sub, .mul, .dup, .drop, .ret => true,
            else => false,
        };
    }
};

/// One decoded instruction.
pub const Inst = union(Opcode) {
    push_i64: i64,
    push_arg: u8,
    add,
    sub,
    mul,
    div,
    rem,
    @"and",
    @"or",
    xor,
    shl,
    shr,
    sar,
    dup,
    drop,
    swap,
    load_local: u8,
    store_local: u8,
    jmp: u16,
    br_if: struct { cond: Cond, target: u16 },
    ret,

    /// Operand-stack depth change of this instruction.
    pub fn stackEffect(i: Inst) i8 {
        return switch (i) {
            .push_i64, .push_arg, .dup, .load_local => 1,
            .add, .sub, .mul, .div, .rem, .@"and", .@"or", .xor, .shl, .shr, .sar, .drop, .ret, .store_local => -1,
            .swap, .jmp => 0,
            .br_if => -2,
        };
    }
    /// Operands this instruction needs on the stack before it runs.
    pub fn pops(i: Inst) u8 {
        return switch (i) {
            .push_i64, .push_arg, .load_local, .jmp => 0,
            .dup, .drop, .ret, .store_local => 1,
            .add, .sub, .mul, .div, .rem, .@"and", .@"or", .xor, .shl, .shr, .sar, .swap, .br_if => 2,
        };
    }
    /// Binary arithmetic opcode (pop b, a; push a op b).
    pub fn isBinary(i: Inst) bool {
        return switch (i) {
            .add, .sub, .mul, .div, .rem, .@"and", .@"or", .xor, .shl, .shr, .sar => true,
            else => false,
        };
    }
};

/// A decoded, validated function.
pub const Function = struct {
    arg_count: u8,
    insts: []const Inst,
    /// Format version (0 or 1).
    version: u8 = version,
    /// v1 (from `decode`): code offset of each instruction and the stack depth before it.
    offsets: []const u16 = &.{},
    depths: []const u8 = &.{},

    /// Frees the slices allocated by `decode`.
    pub fn deinit(f: Function, gpa: std.mem.Allocator) void {
        gpa.free(f.insts);
        if (f.offsets.len != 0) gpa.free(f.offsets);
        if (f.depths.len != 0) gpa.free(f.depths);
    }
    /// Index of the instruction starting at code offset `off` (v1, decoded functions).
    pub fn indexOf(f: Function, off: u16) ?usize {
        return std.sort.binarySearch(u16, f.offsets, off, struct {
            fn cmp(k: u16, x: u16) std.math.Order {
                return std.math.order(k, x);
            }
        }.cmp);
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
    LocalOutOfRange,
    UnknownCondition,
    BadJumpTarget,
    StackMismatch,
    UnreachableCode,
};

/// Human-readable description of a decode error.
pub fn errorMessage(e: Error) []const u8 {
    return switch (e) {
        error.OutOfMemory => "out of memory",
        error.BadMagic => "not a KXBC module (bad magic)",
        error.UnsupportedVersion => "unsupported KXBC version (expected 0 or 1)",
        error.TooManyArgs => "argument count exceeds 4",
        error.Truncated => "module truncated",
        error.TrailingBytes => "bytes after the declared code length",
        error.UnknownOpcode => "unknown opcode (or not available in this version)",
        error.ArgOutOfRange => "push_arg index out of range",
        error.StackUnderflow => "operand stack underflow",
        error.StackTooDeep => "operand stack deeper than 5 (v0) or 64 (v1)",
        error.MissingRet => "function does not end with ret",
        error.CodeAfterRet => "instructions after ret",
        error.LocalOutOfRange => "local index out of range (max 16)",
        error.UnknownCondition => "unknown br_if condition",
        error.BadJumpTarget => "jump target is not an instruction start",
        error.StackMismatch => "different stack depths meet at a jump target",
        error.UnreachableCode => "unreachable instruction",
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
    const ver = bytes[4];
    if (ver > max_version) return error.UnsupportedVersion;
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
    var offs: std.ArrayList(u16) = .empty;
    defer offs.deinit(gpa);
    var pc: usize = 0;
    var depth: u8 = 0;
    var saw_ret = false;
    while (pc < code.len) {
        diag.offset = header_len + pc;
        if (saw_ret and ver == 0) return error.CodeAfterRet;
        const op = std.enums.fromInt(Opcode, code[pc]) orelse return error.UnknownOpcode;
        if (ver == 0 and !op.inV0()) return error.UnknownOpcode;
        const start: u16 = @intCast(pc);
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
            .load_local, .store_local => blk: {
                if (code.len - pc < 1) return error.Truncated;
                const n = code[pc];
                pc += 1;
                if (n >= max_locals) return error.LocalOutOfRange;
                break :blk if (op == .load_local) .{ .load_local = n } else .{ .store_local = n };
            },
            .jmp => blk: {
                if (code.len - pc < 2) return error.Truncated;
                const tg = std.mem.readInt(u16, code[pc..][0..2], .little);
                pc += 2;
                break :blk .{ .jmp = tg };
            },
            .br_if => blk: {
                if (code.len - pc < 3) return error.Truncated;
                const c = std.enums.fromInt(Cond, code[pc]) orelse return error.UnknownCondition;
                const tg = std.mem.readInt(u16, code[pc + 1 ..][0..2], .little);
                pc += 3;
                break :blk .{ .br_if = .{ .cond = c, .target = tg } };
            },
            inline else => |o| o,
        };
        try list.append(gpa, inst);
        if (ver != 0) {
            try offs.append(gpa, start);
            continue;
        }
        if (depth < inst.pops()) return error.StackUnderflow;
        const next: i16 = @as(i16, depth) + inst.stackEffect();
        if (next > max_depth) return error.StackTooDeep;
        depth = @intCast(next);
        if (inst == .ret) saw_ret = true;
    }
    diag.offset = header_len + pc;
    if (ver == 0) {
        if (!saw_ret) return error.MissingRet;
        return .{ .arg_count = arg_count, .insts = try list.toOwnedSlice(gpa) };
    }
    if (list.items.len == 0) return error.MissingRet;
    var f: Function = .{ .arg_count = arg_count, .insts = list.items, .version = ver, .offsets = offs.items };
    const depths = try flow(gpa, f, code.len, diag);
    errdefer gpa.free(depths);
    const offsets = try offs.toOwnedSlice(gpa);
    errdefer gpa.free(offsets);
    // Last, so a failure here leaves `list` to its errdefer and nothing else owned.
    f.insts = try list.toOwnedSlice(gpa);
    f.offsets = offsets;
    f.depths = depths;
    return f;
}

/// v1 data flow: resolves jump targets, computes the stack depth before every instruction,
/// and rejects underflow, overflow, mismatched joins, unreachable code and falling off the end.
fn flow(gpa: std.mem.Allocator, f: Function, code_len: usize, diag: *Diagnostic) Error![]u8 {
    const n = f.insts.len;
    const depth = try gpa.alloc(?u8, n);
    defer gpa.free(depth);
    @memset(depth, null);
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    // Targets first, in program order, so a bad target is reported at its jump.
    for (f.insts, 0..) |inst, i| {
        const tg: ?u16 = switch (inst) {
            .jmp => |x| x,
            .br_if => |x| x.target,
            else => null,
        };
        if (tg) |x| if (f.indexOf(x) == null) {
            diag.offset = header_len + f.offsets[i];
            return error.BadJumpTarget;
        };
    }
    var work: std.ArrayList(usize) = .empty;
    defer work.deinit(gpa);
    depth[0] = 0;
    try work.append(gpa, 0);
    while (work.pop()) |i| {
        const inst = f.insts[i];
        const d = depth[i].?;
        diag.offset = header_len + f.offsets[i];
        if (d < inst.pops()) return error.StackUnderflow;
        const nd_: i16 = @as(i16, d) + inst.stackEffect();
        if (nd_ > max_depth_v1) return error.StackTooDeep;
        const nd: u8 = @intCast(nd_);
        var succ: [2]?usize = .{ null, null };
        switch (inst) {
            .ret => {},
            .jmp => |x| succ[0] = f.indexOf(x).?,
            .br_if => |x| {
                succ[0] = f.indexOf(x.target).?;
                succ[1] = i + 1;
            },
            else => succ[0] = i + 1,
        }
        for (succ) |s_| if (s_) |s| {
            if (s == n) {
                diag.offset = header_len + code_len;
                return error.MissingRet; // falls off the end
            }
            if (depth[s]) |have| {
                if (have != nd) {
                    diag.offset = header_len + f.offsets[s];
                    return error.StackMismatch;
                }
            } else {
                depth[s] = nd;
                try work.append(gpa, s);
            }
        };
    }
    for (depth, 0..) |d, i| {
        out[i] = d orelse {
            diag.offset = header_len + f.offsets[i];
            return error.UnreachableCode;
        };
    }
    return out;
}

/// Errors of the reference interpreter.
pub const EvalError = error{StepLimit};

/// Wrapping integer semantics shared by `eval` and the lowering (division is total).
pub fn binop(i: Inst, a: i64, b: i64) i64 {
    return switch (i) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .div => if (b == 0) 0 else if (b == -1) 0 -% a else @divTrunc(a, b),
        .rem => if (b == 0) a else if (b == -1) 0 else @rem(a, b),
        .@"and" => a & b,
        .@"or" => a | b,
        .xor => a ^ b,
        .shl => a << @as(u6, @truncate(@as(u64, @bitCast(b)))),
        .shr => @bitCast(@as(u64, @bitCast(a)) >> @as(u6, @truncate(@as(u64, @bitCast(b))))),
        .sar => a >> @as(u6, @truncate(@as(u64, @bitCast(b)))),
        else => unreachable,
    };
}

pub fn condHolds(c: Cond, a: i64, b: i64) bool {
    const ua: u64 = @bitCast(a);
    const ub: u64 = @bitCast(b);
    return switch (c) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
        .ult => ua < ub,
        .ule => ua <= ub,
        .ugt => ua > ub,
        .uge => ua >= ub,
    };
}

/// Reference interpreter for a decoded (validated) function of either version.
pub fn eval(f: Function, args: [max_args]i64, max_steps: usize) EvalError!i64 {
    var stack: [max_depth_v1]i64 = undefined;
    var locals: [max_locals]i64 = @splat(0);
    var sp: usize = 0;
    var pc: usize = 0;
    var steps: usize = 0;
    while (true) : (steps += 1) {
        if (steps >= max_steps) return error.StepLimit;
        const inst = f.insts[pc];
        pc += 1;
        switch (inst) {
            .push_i64 => |v| {
                stack[sp] = v;
                sp += 1;
            },
            .push_arg => |n| {
                stack[sp] = args[n];
                sp += 1;
            },
            .dup => {
                stack[sp] = stack[sp - 1];
                sp += 1;
            },
            .drop => sp -= 1,
            .swap => std.mem.swap(i64, &stack[sp - 1], &stack[sp - 2]),
            .load_local => |n| {
                stack[sp] = locals[n];
                sp += 1;
            },
            .store_local => |n| {
                sp -= 1;
                locals[n] = stack[sp];
            },
            .jmp => |tg| pc = f.indexOf(tg).?,
            .br_if => |x| {
                sp -= 2;
                if (condHolds(x.cond, stack[sp], stack[sp + 1])) pc = f.indexOf(x.target).?;
            },
            .ret => return stack[sp - 1],
            else => {
                sp -= 1;
                stack[sp - 1] = binop(inst, stack[sp - 1], stack[sp]);
            },
        }
    }
}

/// Encodes `f` into a fresh module (header + code). Inverse of `decode`.
pub fn encode(gpa: std.mem.Allocator, f: Function) error{ OutOfMemory, CodeTooLarge }![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, magic);
    try out.appendSlice(gpa, &.{ f.version, f.arg_count, 0, 0 });
    for (f.insts) |i| {
        try out.append(gpa, @backingInt(std.meta.activeTag(i)));
        switch (i) {
            .push_i64 => |v| {
                var b: [8]u8 = undefined;
                std.mem.writeInt(i64, &b, v, .little);
                try out.appendSlice(gpa, &b);
            },
            .push_arg, .load_local, .store_local => |n| try out.append(gpa, n),
            .jmp => |tg| {
                var b: [2]u8 = undefined;
                std.mem.writeInt(u16, &b, tg, .little);
                try out.appendSlice(gpa, &b);
            },
            .br_if => |x| {
                var b: [2]u8 = undefined;
                std.mem.writeInt(u16, &b, x.target, .little);
                try out.append(gpa, @backingInt(x.cond));
                try out.appendSlice(gpa, &b);
            },
            else => {},
        }
    }
    const code_len = out.items.len - header_len;
    if (code_len > std.math.maxInt(u16)) return error.CodeTooLarge;
    std.mem.writeInt(u16, out.items[6..8], @intCast(code_len), .little);
    return out.toOwnedSlice(gpa);
}
