//! KXBC v1 lowering: bytecode -> `core/ir.zig` function -> the pyrlyn-jit Compilers
//! (x86-64, AArch64, RV64: linear-scan register allocation, spilling, host C ABI).
//!
//! Every operand-stack slot, local and argument is a mutable IR vreg (`v0..` arguments, then 16
//! locals, then 64 stack slots, then one temporary), so a stack depth of 64 needs no special
//! handling: the allocator spills what does not fit. Basic blocks start at offset 0, at every
//! jump target and after every `jmp`/`br_if`/`ret`. `div`/`rem` are lowered with guards for
//! `b == 0` and `b == -1`, so they never trap and match `bytecode.binop` on every target.
//! v0 functions take the same path when compiled here (the CLI keeps the direct v0 lowering).
const std = @import("std");
const jit = @import("pyrlyn_jit");
const bc = @import("bytecode.zig");
const lower = @import("lower.zig");
const ir = jit.ir;
const opts = jit.build_options;

const VReg = ir.VReg;

/// Errors from `toIr` / `lowerIr`.
pub const Error = error{ OutOfMemory, BackendNotBuilt, CompileFailed };

/// `ir.Func.emit` also rejects over-long argument lists and the reserved vreg; lowering never
/// produces either, so they are reported as `CompileFailed`.
fn emit(g: *ir.Func, b: u32, op: ir.Op) Error!void {
    g.emit(b, op) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.TooManyArgs, error.BadVReg => error.CompileFailed,
    };
}

const Layout = struct {
    args: u32,
    fn local(l: Layout, n: u8) VReg {
        return l.args + n;
    }
    fn slot(l: Layout, d: usize) VReg {
        return l.args + bc.max_locals + @as(u32, @intCast(d));
    }
    fn tmp(l: Layout) VReg {
        return l.args + bc.max_locals + bc.max_depth_v1;
    }
    fn tmp2(l: Layout) VReg {
        return l.tmp() + 1;
    }
};

fn irCond(c: bc.Cond) ir.CondCode {
    return @fromBackingInt(@intCast(@backingInt(c)));
}

/// Stack depth before each instruction (v1 functions carry it; v0 is straight-line).
fn depthsOf(gpa: std.mem.Allocator, f: bc.Function) ![]u8 {
    const out = try gpa.alloc(u8, f.insts.len);
    if (f.depths.len == f.insts.len) {
        @memcpy(out, f.depths);
        return out;
    }
    var d: i16 = 0;
    for (f.insts, 0..) |inst, i| {
        out[i] = @intCast(d);
        d += inst.stackEffect();
    }
    return out;
}

/// Builds the IR function for `f` (decoded and validated by `bytecode.decode`).
pub fn toIr(gpa: std.mem.Allocator, f: bc.Function) Error!ir.Func {
    var g = ir.Func.init(gpa);
    errdefer g.deinit();
    g.params = f.arg_count;
    const L: Layout = .{ .args = f.arg_count };
    const depth = try depthsOf(gpa, f);
    defer gpa.free(depth);
    const n = f.insts.len;

    // Leaders and their blocks.
    const block_of = try gpa.alloc(?u32, n + 1);
    defer gpa.free(block_of);
    @memset(block_of, null);
    const leader = try gpa.alloc(bool, n);
    defer gpa.free(leader);
    @memset(leader, false);
    leader[0] = true;
    for (f.insts, 0..) |inst, i| switch (inst) {
        .jmp => |tg| {
            leader[f.indexOf(tg).?] = true;
            if (i + 1 < n) leader[i + 1] = true;
        },
        .br_if => |x| {
            leader[f.indexOf(x.target).?] = true;
            if (i + 1 < n) leader[i + 1] = true;
        },
        .ret => if (i + 1 < n) {
            leader[i + 1] = true;
        },
        else => {},
    };
    const A = g.alloc();
    const entry = try g.addBlock("entry");
    for (0..n) |i| if (leader[i]) {
        block_of[i] = try g.addBlock(try std.fmt.allocPrint(A, "b{d}", .{i}));
    };
    for (0..bc.max_locals) |k| try emit(&g, entry, .{ .@"const" = .{ .dst = L.local(@intCast(k)), .imm = 0 } });
    try emit(&g, entry, .{ .br = block_of[0].? });

    var cur: u32 = entry;
    var extra: u32 = 0;
    for (f.insts, 0..) |inst, i| {
        if (block_of[i]) |b| {
            // Fall-through from the previous instruction into a leader.
            if (i != 0 and !isTerminator(f.insts[i - 1])) try emit(&g, cur, .{ .br = b });
            cur = b;
        }
        const d: usize = depth[i];
        switch (inst) {
            .push_i64 => |v| try emit(&g, cur, .{ .@"const" = .{ .dst = L.slot(d), .imm = v } }),
            .push_arg => |a| try emit(&g, cur, .{ .copy = .{ .dst = L.slot(d), .src = a } }),
            .dup => try emit(&g, cur, .{ .copy = .{ .dst = L.slot(d), .src = L.slot(d - 1) } }),
            .drop => {},
            .swap => {
                try emit(&g, cur, .{ .copy = .{ .dst = L.tmp(), .src = L.slot(d - 1) } });
                try emit(&g, cur, .{ .copy = .{ .dst = L.slot(d - 1), .src = L.slot(d - 2) } });
                try emit(&g, cur, .{ .copy = .{ .dst = L.slot(d - 2), .src = L.tmp() } });
            },
            .load_local => |k| try emit(&g, cur, .{ .copy = .{ .dst = L.slot(d), .src = L.local(k) } }),
            .store_local => |k| try emit(&g, cur, .{ .copy = .{ .dst = L.local(k), .src = L.slot(d - 1) } }),
            .jmp => |tg| try emit(&g, cur, .{ .br = block_of[f.indexOf(tg).?].? }),
            .br_if => |x| try emit(&g, cur, .{ .cbr = .{ .cond = irCond(x.cond), .a = L.slot(d - 2), .b = L.slot(d - 1), .then = block_of[f.indexOf(x.target).?].?, .@"else" = block_of[i + 1].? } }),
            .ret => try emit(&g, cur, .{ .ret = L.slot(d - 1) }),
            .div, .rem => {
                // Guards make division total: b == 0 -> (0 | a), b == -1 -> (-a | 0), else sdiv/srem.
                const a = L.slot(d - 2);
                const b = L.slot(d - 1);
                const is_div = inst == .div;
                const name = struct {
                    fn f_(al: std.mem.Allocator, i_: usize, k: *u32, s: []const u8) ![]const u8 {
                        k.* += 1;
                        return std.fmt.allocPrint(al, "b{d}.{s}{d}", .{ i_, s, k.* });
                    }
                }.f_;
                const zb = try g.addBlock(try name(A, i, &extra, "zero"));
                const nb_chk = try g.addBlock(try name(A, i, &extra, "chk"));
                const nb = try g.addBlock(try name(A, i, &extra, "neg"));
                const db = try g.addBlock(try name(A, i, &extra, "div"));
                const join = try g.addBlock(try name(A, i, &extra, "join"));
                try emit(&g, cur, .{ .@"const" = .{ .dst = L.tmp(), .imm = 0 } });
                try emit(&g, cur, .{ .cbr = .{ .cond = .eq, .a = b, .b = L.tmp(), .then = zb, .@"else" = nb_chk } });
                if (is_div) try emit(&g, zb, .{ .@"const" = .{ .dst = a, .imm = 0 } }); // rem: a % 0 = a
                try emit(&g, zb, .{ .br = join });
                try emit(&g, nb_chk, .{ .@"const" = .{ .dst = L.tmp(), .imm = -1 } });
                try emit(&g, nb_chk, .{ .cbr = .{ .cond = .eq, .a = b, .b = L.tmp(), .then = nb, .@"else" = db } });
                if (is_div) {
                    try emit(&g, nb, .{ .@"const" = .{ .dst = L.tmp2(), .imm = 0 } });
                    try emit(&g, nb, .{ .arith = .{ .op = .sub, .dst = a, .a = L.tmp2(), .b = a } });
                } else try emit(&g, nb, .{ .@"const" = .{ .dst = a, .imm = 0 } });
                try emit(&g, nb, .{ .br = join });
                try emit(&g, db, .{ .arith = .{ .op = if (is_div) .sdiv else .srem, .dst = a, .a = a, .b = b } });
                try emit(&g, db, .{ .br = join });
                cur = join;
            },
            else => {
                const op: ir.Arith = switch (inst) {
                    .add => .add,
                    .sub => .sub,
                    .mul => .mul,
                    .@"and" => .@"and",
                    .@"or" => .@"or",
                    .xor => .xor,
                    .shl => .shl,
                    .shr => .shr,
                    .sar => .sar,
                    else => unreachable,
                };
                try emit(&g, cur, .{ .arith = .{ .op = op, .dst = L.slot(d - 2), .a = L.slot(d - 2), .b = L.slot(d - 1) } });
            },
        }
    }
    return g;
}

fn isTerminator(i: bc.Inst) bool {
    return i == .jmp or i == .br_if or i == .ret;
}

/// Lowers `f` through the IR Compiler for `t` into `cb` (finalized).
pub fn lowerIr(gpa: std.mem.Allocator, t: lower.Target, f: bc.Function, cb: *jit.CodeBuffer) Error!void {
    return lowerIrOpt(gpa, t, f, cb, .o0);
}

/// `lowerIr` with the IR optimizer (`jit.opt`) run at `level` before code generation; `.o2`
/// also vectorizes loops for the target's vector unit (AArch64 NEON, x86-64 SSE).
pub fn lowerIrOpt(gpa: std.mem.Allocator, t: lower.Target, f: bc.Function, cb: *jit.CodeBuffer, level: jit.opt.Level) Error!void {
    var g = try toIr(gpa, f);
    defer g.deinit();
    if (level != .o0) {
        const vector: jit.opt.VectorTarget = if (level != .o2) .none else switch (t) {
            .x86_64 => .x86_64,
            .aarch64 => .aarch64,
            .riscv64 => .none,
        };
        _ = jit.opt.optimize(gpa, &g, .{ .level = level, .vector = vector }) catch |e|
            return if (e == error.OutOfMemory) error.OutOfMemory else error.CompileFailed;
    }
    switch (t) {
        .x86_64 => if (comptime opts.x86_64) {
            _ = jit.x86_64_compiler.compile(gpa, &g, cb, .{}) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.CompileFailed;
        } else return error.BackendNotBuilt,
        .aarch64 => if (comptime opts.aarch64) {
            _ = jit.aarch64_compiler.compile(gpa, &g, cb, .{}) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.CompileFailed;
        } else return error.BackendNotBuilt,
        .riscv64 => _ = jit.riscv_compiler.compile(gpa, &g, cb, .{}) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.CompileFailed,
    }
}
