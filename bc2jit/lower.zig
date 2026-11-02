//! Lowers a decoded KXBC function to machine code with the pyrlyn-jit assemblers.
//!
//! Register model: operand-stack slot `i` lives in a fixed caller-saved register `slots[i]`
//! (no spills; `bytecode.max_depth` bounds the depth). Arguments stay in their ABI registers
//! and `push_arg` copies them. `ret` moves slot 0 of the final stack into the return register.
//! Calling conventions: x86-64 SysV, AAPCS64, RV64 LP64 (integer arguments only).
const std = @import("std");
const builtin = @import("builtin");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");

const CodeBuffer = jit.CodeBuffer;
const opts = jit.build_options;

/// Target architectures selectable with `--arch`.
pub const Target = enum {
    x86_64,
    aarch64,
    riscv64,

    /// Value accepted by `--arch` for this target.
    pub fn flagName(t: Target) []const u8 {
        return switch (t) {
            .x86_64 => "x86-64",
            .aarch64 => "aarch64",
            .riscv64 => "riscv",
        };
    }
    /// Parses an `--arch` value; null for anything else.
    pub fn fromFlag(s: []const u8) ?Target {
        inline for (std.meta.tags(Target)) |t| {
            if (std.mem.eql(u8, s, t.flagName())) return t;
        }
        return null;
    }
    /// Whether this target's backend was compiled into pyrlyn-jit (`-Dtargets=`).
    /// RISC-V is always built; x86-64 and AArch64 follow the build options.
    pub fn isBuilt(t: Target) bool {
        return switch (t) {
            .x86_64 => opts.x86_64,
            .aarch64 => opts.aarch64,
            .riscv64 => true,
        };
    }
    /// The target matching the machine this program runs on, if any.
    pub fn host() ?Target {
        return switch (builtin.cpu.arch) {
            .x86_64 => .x86_64,
            .aarch64 => .aarch64,
            .riscv64 => .riscv64,
            else => null,
        };
    }
    /// Whether code for this target can be executed in this process: target == host and the
    /// host ABI matches the one the lowering uses (Windows x64 is excluded: it is not SysV).
    pub fn canExecute(t: Target) bool {
        return host() == t and !(t == .x86_64 and builtin.os.tag == .windows);
    }
};

/// Comma-separated `--arch` values, for messages.
pub const allowed_flags = "x86-64, aarch64, riscv";

/// Lowering failures.
pub const Error = error{BackendNotBuilt} || (if (opts.x86_64) jit.x86_64.Error else error{}) ||
    (if (opts.aarch64) jit.aarch64.Error else error{}) || jit.riscv.Error;

/// Appends machine code for `f` to `cb` and finalizes it (labels resolved, ready for
/// `JitRuntime.add` or for writing out as raw bytes).
pub fn lower(t: Target, f: bc.Function, cb: *CodeBuffer) Error!void {
    switch (t) {
        .x86_64 => if (comptime opts.x86_64) try emit(X86, jit.x86_64.Assembler.init(cb), f) else return error.BackendNotBuilt,
        .aarch64 => if (comptime opts.aarch64) try emit(A64, jit.aarch64.Assembler.init(cb), f) else return error.BackendNotBuilt,
        .riscv64 => try emit(Rv, jit.riscv.Assembler.init(cb), f),
    }
    try cb.finalize();
}

fn emit(comptime M: type, a: M.Asm, f: bc.Function) !void {
    var depth: usize = 0;
    for (f.insts) |inst| {
        switch (inst) {
            .push_i64 => |v| try M.movImm(a, M.slots[depth], v),
            .push_arg => |n| try M.mov(a, M.slots[depth], M.args[n]),
            .add => try M.add(a, M.slots[depth - 2], M.slots[depth - 1]),
            .sub => try M.sub(a, M.slots[depth - 2], M.slots[depth - 1]),
            .mul => try M.mul(a, M.slots[depth - 2], M.slots[depth - 1]),
            .dup => try M.mov(a, M.slots[depth], M.slots[depth - 1]),
            .drop => {},
            .ret => {
                try M.mov(a, M.ret_reg, M.slots[depth - 1]);
                try a.ret();
            },
        }
        depth = @intCast(@as(isize, @intCast(depth)) + inst.stackEffect());
    }
}

/// x86-64 SysV mapping. Slot registers avoid the argument registers rdi/rsi/rdx/rcx.
pub const X86 = if (opts.x86_64) struct {
    const G = jit.x86_64.Gp;
    pub const Asm = jit.x86_64.Assembler;
    pub const args = [bc.max_args]G{ .rdi, .rsi, .rdx, .rcx };
    pub const slots = [bc.max_depth]G{ .rax, .r10, .r11, .r8, .r9 };
    pub const ret_reg: G = .rax;
    fn movImm(a: Asm, d: G, v: i64) !void {
        try a.movRI(d, v);
    }
    fn mov(a: Asm, d: G, s: G) !void {
        if (d != s) try a.movRR(d, s);
    }
    fn add(a: Asm, d: G, s: G) !void {
        try a.aluRR(.add, d, s);
    }
    fn sub(a: Asm, d: G, s: G) !void {
        try a.aluRR(.sub, d, s);
    }
    fn mul(a: Asm, d: G, s: G) !void {
        try a.imulRR(d, s);
    }
} else void;

/// AAPCS64 mapping. Slot registers are the temporaries x9..x13.
pub const A64 = if (opts.aarch64) struct {
    const R = jit.aarch64.X;
    pub const Asm = jit.aarch64.Assembler;
    pub const args = [bc.max_args]R{ .x0, .x1, .x2, .x3 };
    pub const slots = [bc.max_depth]R{ .x9, .x10, .x11, .x12, .x13 };
    pub const ret_reg: R = .x0;
    fn movImm(a: Asm, d: R, v: i64) !void {
        try a.movImm(d, @bitCast(v));
    }
    fn mov(a: Asm, d: R, s: R) !void {
        if (d != s) try a.mov(d, s);
    }
    fn add(a: Asm, d: R, s: R) !void {
        try a.add(d, d, s);
    }
    fn sub(a: Asm, d: R, s: R) !void {
        try a.sub(d, d, s);
    }
    fn mul(a: Asm, d: R, s: R) !void {
        try a.mul(d, d, s);
    }
} else void;

/// RV64 LP64 mapping. Slot registers are the temporaries t0..t4.
pub const Rv = struct {
    const R = jit.riscv.X;
    pub const Asm = jit.riscv.Assembler;
    pub const args = [bc.max_args]R{ .a0, .a1, .a2, .a3 };
    pub const slots = [bc.max_depth]R{ .t0, .t1, .t2, .t3, .t4 };
    pub const ret_reg: R = .a0;
    fn movImm(a: Asm, d: R, v: i64) !void {
        try a.li(d, v);
    }
    fn mov(a: Asm, d: R, s: R) !void {
        if (d != s) try a.mv(d, s);
    }
    fn add(a: Asm, d: R, s: R) !void {
        try a.add(d, d, s);
    }
    fn sub(a: Asm, d: R, s: R) !void {
        try a.sub(d, d, s);
    }
    fn mul(a: Asm, d: R, s: R) !void {
        try a.mul(d, d, s);
    }
};
