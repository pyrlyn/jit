//! Minimal compiler IR (docs/ROADMAP.md §2): i64 values in virtual registers, functions of
//! basic blocks, three-address instructions, a text parser, a verifier, a printer and a
//! reference interpreter (used for differential tests against generated code).
//!
//! Values are mutable virtual registers (`v0`, `v1`, ...) rather than SSA with block
//! parameters: the linear-scan allocator works on def/use sets, so no phi lowering is needed.
//! Parameters are `v0 .. v(params-1)`.
//!
//! Text format (one instruction per line, `#` starts a comment):
//!   func <name> <params>
//!   <label>:
//!     vD = const <int>          vD = copy vA
//!     vD = add|sub|mul|and|or|xor|shl|shr|sar|sdiv|srem vA vB
//!     vD = load vA <disp>       store vA <disp> vB         (64-bit memory at vA+disp)
//!     vD = call <addr> vA ...   (up to 16 args, C calling convention)
//!     br <label>                cbr <cond> vA vB <then> <else>
//!     ret vA
//!
//! Block parameters (SSA style, optional): a block header `name(vP vQ):` declares
//! parameters; edges pass arguments with `br name vA vB` and `cbr <cond> vA vB then:vC else:vD:vE`
//! (`:`-separated argument lists). Arguments are assigned in parallel on the edge.
//! `verifySsa` checks single definitions; `lowerBlockParams` rewrites edges into copies
//! (splitting `cbr` edges that carry arguments) so the compilers see plain mutable vregs.
//!   conds: eq ne lt le gt ge ult ule ugt uge
//!
//! Typed values (f32/f64 scalars, 128-bit vectors v4i32/v2i64/v4f32/v2f64). Untyped ops work on
//! i64; a vreg's type comes from its definition (`typeCheck` infers and checks them):
//!   vD = fconst.<f32|f64> <float>          vD = <add|sub|mul|div|min|max|and|or|xor>.<type> vA vB
//!   vD = fcmp.<f32|f64> <eq|ne|lt|le|gt|ge> vA vB   (i64 0/1; ordered, false on NaN except ne)
//!   vD = cvt.<to>.<from> vA        (i64 <-> f32/f64, f32 <-> f64; float -> i64 truncates)
//!   vD = splat.<vector> vA         vD = extract.<vector> vA <lane>   (lane scalar: i64/f32/f64)
//!   vD = load.<type> vA <disp>     store.<type> vA <disp> vB
//! `div`/`min`/`max` are float-only; `and`/`or`/`xor` integer-vector-only. Vector integer lanes
//! wrap; `splat.v4i32` truncates the i64, `extract.v4i32` sign-extends.
const std = @import("std");

pub const Type = enum {
    i64,
    f32,
    f64,
    v4i32,
    v2i64,
    v4f32,
    v2f64,
    pub fn isFloat(tp: Type) bool {
        return tp == .f32 or tp == .f64 or tp == .v4f32 or tp == .v2f64;
    }
    pub fn isVector(tp: Type) bool {
        return @backingInt(tp) >= @backingInt(Type.v4i32);
    }
    /// Scalar type of one lane (`i64` for integer lanes).
    pub fn lane(tp: Type) Type {
        return switch (tp) {
            .v4i32, .v2i64 => .i64,
            .v4f32 => .f32,
            .v2f64 => .f64,
            else => tp,
        };
    }
    pub fn lanes(tp: Type) u8 {
        return switch (tp) {
            .v4i32, .v4f32 => 4,
            .v2i64, .v2f64 => 2,
            else => 1,
        };
    }
    /// Size in memory (`load`/`store`).
    pub fn size(tp: Type) u8 {
        return switch (tp) {
            .f32 => 4,
            .i64, .f64 => 8,
            else => 16,
        };
    }
};
pub const TArith = enum { add, sub, mul, div, min, max, @"and", @"or", xor };
pub const FCond = enum { eq, ne, lt, le, gt, ge };
/// Raw value of a vreg in the interpreter (scalars in the low bits).
pub const Val = u128;

pub const VReg = u32;
/// Maximum call arguments and function parameters (extra ones are passed on the stack).
pub const max_args = 16;
/// `shl/shr/sar` mask the count to 6 bits; `sdiv/srem` truncate toward zero and are
/// undefined (trap in generated code) for a zero divisor or `minInt(i64) / -1`.
pub const Arith = enum { add, sub, mul, @"and", @"or", xor, shl, shr, sar, sdiv, srem };
pub const CondCode = enum { eq, ne, lt, le, gt, ge, ult, ule, ugt, uge };

pub const Op = union(enum) {
    @"const": struct { dst: VReg, imm: i64 },
    copy: struct { dst: VReg, src: VReg },
    arith: struct { op: Arith, dst: VReg, a: VReg, b: VReg },
    load: struct { dst: VReg, base: VReg, disp: i32, ty: Type = .i64 },
    store: struct { base: VReg, disp: i32, src: VReg, ty: Type = .i64 },
    /// Float constant: `bits` holds the f32 (low 32 bits) or f64 bit pattern.
    fconst: struct { dst: VReg, ty: Type, bits: u64 },
    /// Typed arithmetic on f32/f64 or vector values (both operands and the result of `ty`).
    typed: struct { op: TArith, ty: Type, dst: VReg, a: VReg, b: VReg },
    fcmp: struct { cond: FCond, ty: Type, dst: VReg, a: VReg, b: VReg },
    cvt: struct { to: Type, from: Type, dst: VReg, src: VReg },
    splat: struct { ty: Type, dst: VReg, src: VReg },
    extract: struct { ty: Type, dst: VReg, src: VReg, lane: u8 },
    call: struct { dst: VReg, addr: usize, args: []const VReg },
    br: u32,
    cbr: struct { cond: CondCode, a: VReg, b: VReg, then: u32, @"else": u32 },
    /// `br` with block arguments.
    bra: struct { target: u32, args: []const VReg },
    /// `cbr` with block arguments on each edge.
    cbra: struct { cond: CondCode, a: VReg, b: VReg, then: u32, @"else": u32, then_args: []const VReg, else_args: []const VReg },
    ret: VReg,

    pub fn isTerminator(o: Op) bool {
        return switch (o) {
            .br, .cbr, .bra, .cbra, .ret => true,
            else => false,
        };
    }
    pub fn def(o: Op) ?VReg {
        return switch (o) {
            inline .@"const", .copy, .arith, .load, .call, .fconst, .typed, .fcmp, .cvt, .splat, .extract => |x| x.dst,
            else => null,
        };
    }
    /// Writes used vregs into `buf`, returns the slice.
    pub fn uses(o: Op, buf: *[max_args]VReg) []const VReg {
        return switch (o) {
            .@"const", .br, .fconst => buf[0..0],
            .typed => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                break :blk buf[0..2];
            },
            .fcmp => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                break :blk buf[0..2];
            },
            inline .cvt, .splat, .extract => |x| blk: {
                buf[0] = x.src;
                break :blk buf[0..1];
            },
            .copy => |x| blk: {
                buf[0] = x.src;
                break :blk buf[0..1];
            },
            .arith => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                break :blk buf[0..2];
            },
            .load => |x| blk: {
                buf[0] = x.base;
                break :blk buf[0..1];
            },
            .store => |x| blk: {
                buf[0] = x.base;
                buf[1] = x.src;
                break :blk buf[0..2];
            },
            .call => |x| blk: {
                for (x.args, 0..) |a, i| buf[i] = a;
                break :blk buf[0..x.args.len];
            },
            .cbr => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                break :blk buf[0..2];
            },
            .bra => |x| blk: {
                for (x.args, 0..) |a, i| buf[i] = a;
                break :blk buf[0..x.args.len];
            },
            .cbra => |x| blk: {
                buf[0] = x.a;
                buf[1] = x.b;
                for (x.then_args, 2..) |a, i| buf[i] = a;
                for (x.else_args, 2 + x.then_args.len..) |a, i| buf[i] = a;
                break :blk buf[0 .. 2 + x.then_args.len + x.else_args.len];
            },
            .ret => |v| blk: {
                buf[0] = v;
                break :blk buf[0..1];
            },
        };
    }
};

pub const Block = struct {
    name: []const u8,
    ops: std.ArrayList(Op) = .empty,
    /// Block parameters (assigned by incoming `bra`/`cbra` edges).
    params: []const VReg = &.{},
};

pub const Func = struct {
    arena: std.heap.ArenaAllocator,
    name: []const u8 = "",
    params: u32 = 0,
    num_vregs: u32 = 0,
    blocks: std.ArrayList(Block) = .empty,

    pub fn init(gpa: std.mem.Allocator) Func {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }
    pub fn deinit(f: *Func) void {
        f.arena.deinit();
    }
    pub fn alloc(f: *Func) std.mem.Allocator {
        return f.arena.allocator();
    }

    // ---- IrBuilder-style API ----------------------------------------------------------
    pub fn newVReg(f: *Func) VReg {
        f.num_vregs += 1;
        return f.num_vregs - 1;
    }
    pub fn addBlock(f: *Func, name: []const u8) !u32 {
        try f.blocks.append(f.alloc(), .{ .name = try f.alloc().dupe(u8, name) });
        return @intCast(f.blocks.items.len - 1);
    }
    pub fn emit(f: *Func, block: u32, op: Op) !void {
        // `uses` fills a `max_args` buffer, and `num_vregs = v + 1` must not overflow.
        switch (op) {
            .call => |c| if (c.args.len > max_args) return error.TooManyArgs,
            .bra => |c| if (c.args.len > max_args) return error.TooManyArgs,
            .cbra => |c| if (2 + c.then_args.len + c.else_args.len > max_args) return error.TooManyArgs,
            else => {},
        }
        var buf: [max_args]VReg = undefined;
        for (op.uses(&buf)) |u| if (u == std.math.maxInt(VReg)) return error.BadVReg;
        if (op.def()) |d| if (d == std.math.maxInt(VReg)) return error.BadVReg;
        try f.blocks.items[block].ops.append(f.alloc(), op);
        if (op.def()) |d| f.num_vregs = @max(f.num_vregs, d + 1);
        for (op.uses(&buf)) |u| f.num_vregs = @max(f.num_vregs, u + 1);
    }

    pub const VerifyError = error{ EmptyFunction, MissingTerminator, TerminatorNotLast, BadBlock, BadVReg, TooManyArgs, ArgCountMismatch, MultipleDefs };
    /// Structural checks: every block ends with exactly one terminator, branch targets exist,
    /// vregs are in range, calls and edges carry at most `max_args` arguments.
    pub fn verify(f: *const Func) VerifyError!void {
        if (f.blocks.items.len == 0) return error.EmptyFunction;
        const nb = f.blocks.items.len;
        for (f.blocks.items) |b| {
            if (b.ops.items.len == 0 or !b.ops.items[b.ops.items.len - 1].isTerminator()) return error.MissingTerminator;
            // One switch per op (terminator position, branch targets and argument counts, vreg
            // range), checked in that order as before.
            const n = f.num_vregs;
            const last = b.ops.items.len - 1;
            const C = struct {
                inline fn all(vs: []const VReg, n_: u32) bool {
                    for (vs) |v| if (v >= n_) return false;
                    return true;
                }
            };
            for (b.ops.items, 0..) |*o, i| switch (o.*) {
                inline .@"const", .fconst => |x| if (x.dst >= n) return error.BadVReg,
                inline .copy, .cvt, .splat, .extract => |x| if (x.src >= n or x.dst >= n) return error.BadVReg,
                inline .arith, .typed, .fcmp => |x| if (x.a >= n or x.b >= n or x.dst >= n) return error.BadVReg,
                .load => |x| if (x.base >= n or x.dst >= n) return error.BadVReg,
                .store => |x| if (x.base >= n or x.src >= n) return error.BadVReg,
                .call => |c| {
                    if (c.args.len > max_args) return error.TooManyArgs;
                    if (!C.all(c.args, n) or c.dst >= n) return error.BadVReg;
                },
                .br => |tg| {
                    if (i != last) return error.TerminatorNotLast;
                    if (tg >= nb) return error.BadBlock;
                    if (f.blocks.items[tg].params.len != 0) return error.ArgCountMismatch;
                },
                .cbr => |c| {
                    if (i != last) return error.TerminatorNotLast;
                    if (c.then >= nb or c.@"else" >= nb) return error.BadBlock;
                    if (f.blocks.items[c.then].params.len != 0 or f.blocks.items[c.@"else"].params.len != 0) return error.ArgCountMismatch;
                    if (c.a >= n or c.b >= n) return error.BadVReg;
                },
                .bra => |c| {
                    if (i != last) return error.TerminatorNotLast;
                    if (c.target >= nb) return error.BadBlock;
                    if (c.args.len > max_args) return error.TooManyArgs;
                    if (c.args.len != f.blocks.items[c.target].params.len) return error.ArgCountMismatch;
                    if (!C.all(c.args, n)) return error.BadVReg;
                },
                .cbra => |c| {
                    if (i != last) return error.TerminatorNotLast;
                    if (c.then >= nb or c.@"else" >= nb) return error.BadBlock;
                    if (2 + c.then_args.len + c.else_args.len > max_args) return error.TooManyArgs;
                    if (c.then_args.len != f.blocks.items[c.then].params.len or
                        c.else_args.len != f.blocks.items[c.@"else"].params.len) return error.ArgCountMismatch;
                    if (c.a >= n or c.b >= n or !C.all(c.then_args, n) or !C.all(c.else_args, n)) return error.BadVReg;
                },
                .ret => |v| {
                    if (i != last) return error.TerminatorNotLast;
                    if (v >= n) return error.BadVReg;
                },
            };
        }
        for (f.blocks.items) |b| for (b.params) |p| if (p >= f.num_vregs) return error.BadVReg;
        if (f.params > f.num_vregs and f.params > 0) return error.BadVReg;
    }

    /// `verify` plus SSA form: every vreg is defined once (function parameters, block
    /// parameters and instruction results count as definitions).
    pub fn verifySsa(f: *const Func, gpa: std.mem.Allocator) (VerifyError || error{OutOfMemory})!void {
        try f.verify();
        const seen = try gpa.alloc(bool, f.num_vregs);
        defer gpa.free(seen);
        @memset(seen, false);
        for (0..f.params) |i| seen[i] = true;
        for (f.blocks.items) |b| {
            for (b.params) |p| {
                if (seen[p]) return error.MultipleDefs;
                seen[p] = true;
            }
            for (b.ops.items) |o| if (o.def()) |d| {
                if (seen[d]) return error.MultipleDefs;
                seen[d] = true;
            };
        }
    }

    /// True if the function only uses untyped (i64) operations (what the compilers support).
    pub fn isIntOnly(f: *const Func) bool {
        for (f.blocks.items) |b| for (b.ops.items) |o| switch (o) {
            .fconst, .typed, .fcmp, .cvt, .splat, .extract => return false,
            .load => |x| if (x.ty != .i64) return false,
            .store => |x| if (x.ty != .i64) return false,
            else => {},
        };
        return true;
    }

    pub const TypeError = error{ TypeMismatch, OutOfMemory };
    /// Infers the type of every vreg (parameters i64, `copy`/block parameters from their
    /// sources) and checks operand types. Returns `types[v]` (caller frees with `gpa`).
    pub fn typeCheck(f: *const Func, gpa: std.mem.Allocator) TypeError![]Type {
        const ty = try gpa.alloc(?Type, @max(f.num_vregs, 1));
        defer gpa.free(ty);
        @memset(ty, null);
        for (0..f.params) |i| ty[i] = .i64;
        const set = struct {
            fn f_(t_: []?Type, v: VReg, want: Type) error{TypeMismatch}!bool {
                if (t_[v]) |have| {
                    if (have != want) return error.TypeMismatch;
                    return false;
                }
                t_[v] = want;
                return true;
            }
        }.f_;
        // Defs with fixed types in one pass; copies / block arguments whose source is not
        // typed yet go to a worklist that is iterated to a fixpoint (types only flow from
        // source to destination, and every constraint is an equality, so the order does not
        // change the result or whether there is a mismatch).
        const Pending = struct { b: u32, i: u32 };
        var work: std.ArrayList(Pending) = .empty;
        defer work.deinit(gpa);
        const flow = struct {
            /// Propagates what it can; true if some source is still untyped.
            fn edge(t_: []?Type, fb: []const Block, args: []const VReg, target: u32, ch: *bool) error{TypeMismatch}!bool {
                var open = false;
                for (args, fb[target].params) |a, p| if (t_[a]) |at| {
                    if (try set(t_, p, at)) ch.* = true;
                } else {
                    open = true;
                };
                return open;
            }
            fn op(t_: []?Type, fb: []const Block, o: *const Op, ch: *bool) error{TypeMismatch}!bool {
                return switch (o.*) {
                    .copy => |x| if (t_[x.src]) |st| blk: {
                        if (try set(t_, x.dst, st)) ch.* = true;
                        break :blk false;
                    } else true,
                    .bra => |x| try edge(t_, fb, x.args, x.target, ch),
                    .cbra => |x| blk: {
                        const a = try edge(t_, fb, x.then_args, x.then, ch);
                        const b = try edge(t_, fb, x.else_args, x.@"else", ch);
                        break :blk a or b;
                    },
                    else => false,
                };
            }
        };
        var ch = false;
        for (f.blocks.items, 0..) |b, bi| for (b.ops.items, 0..) |*o, oi| {
            const r: ?struct { VReg, Type } = switch (o.*) {
                .@"const" => |x| .{ x.dst, .i64 },
                .arith => |x| .{ x.dst, .i64 },
                .call => |x| .{ x.dst, .i64 },
                .load => |x| .{ x.dst, x.ty },
                .fconst => |x| .{ x.dst, x.ty },
                .typed => |x| .{ x.dst, x.ty },
                .fcmp => |x| .{ x.dst, .i64 },
                .cvt => |x| .{ x.dst, x.to },
                .splat => |x| .{ x.dst, x.ty },
                .extract => |x| .{ x.dst, x.ty.lane() },
                .copy, .bra, .cbra => blk: {
                    if (try flow.op(ty, f.blocks.items, o, &ch)) try work.append(gpa, .{ .b = @intCast(bi), .i = @intCast(oi) });
                    break :blk null;
                },
                else => null,
            };
            if (r) |d| _ = try set(ty, d[0], d[1]);
        };
        // Another round only if the previous one typed some vreg.
        ch = true;
        while (ch and work.items.len != 0) {
            ch = false;
            var keep: usize = 0;
            for (work.items) |w| if (try flow.op(ty, f.blocks.items, &f.blocks.items[w.b].ops.items[w.i], &ch)) {
                work.items[keep] = w;
                keep += 1;
            };
            work.items.len = keep;
        }
        const out = try gpa.alloc(Type, @max(f.num_vregs, 1));
        errdefer gpa.free(out);
        for (ty, 0..) |t_, i| out[i] = t_ orelse .i64; // never defined: reads as i64 0
        const T = struct {
            fn is(o_: []const Type, v: VReg, want: Type) error{TypeMismatch}!void {
                if (o_[v] != want) return error.TypeMismatch;
            }
        };
        for (f.blocks.items) |b| for (b.ops.items) |o| switch (o) {
            .arith => |x| {
                try T.is(out, x.a, .i64);
                try T.is(out, x.b, .i64);
            },
            .cbr => |x| {
                try T.is(out, x.a, .i64);
                try T.is(out, x.b, .i64);
            },
            .cbra => |x| {
                try T.is(out, x.a, .i64);
                try T.is(out, x.b, .i64);
            },
            .ret => |v| try T.is(out, v, .i64),
            .call => |x| for (x.args) |a| try T.is(out, a, .i64),
            .load => |x| try T.is(out, x.base, .i64),
            .store => |x| {
                try T.is(out, x.base, .i64);
                try T.is(out, x.src, x.ty);
            },
            .fconst => |x| if (x.ty != .f32 and x.ty != .f64) return error.TypeMismatch,
            .typed => |x| {
                if (x.ty == .i64) return error.TypeMismatch;
                const fl = x.ty.isFloat();
                switch (x.op) {
                    .div, .min, .max => if (!fl) return error.TypeMismatch,
                    .@"and", .@"or", .xor => if (fl) return error.TypeMismatch,
                    else => {},
                }
                try T.is(out, x.a, x.ty);
                try T.is(out, x.b, x.ty);
            },
            .fcmp => |x| {
                if (x.ty != .f32 and x.ty != .f64) return error.TypeMismatch;
                try T.is(out, x.a, x.ty);
                try T.is(out, x.b, x.ty);
            },
            .cvt => |x| {
                const ok = switch (x.to) {
                    .i64 => x.from == .f32 or x.from == .f64,
                    .f32 => x.from == .i64 or x.from == .f64,
                    .f64 => x.from == .i64 or x.from == .f32,
                    else => false,
                };
                if (!ok) return error.TypeMismatch;
                try T.is(out, x.src, x.from);
            },
            .splat => |x| {
                if (!x.ty.isVector()) return error.TypeMismatch;
                try T.is(out, x.src, x.ty.lane());
            },
            .extract => |x| {
                if (!x.ty.isVector() or x.lane >= x.ty.lanes()) return error.TypeMismatch;
                try T.is(out, x.src, x.ty);
            },
            else => {},
        };
        return out;
    }

    pub fn hasBlockParams(f: *const Func) bool {
        for (f.blocks.items) |b| if (b.params.len != 0) return true;
        return false;
    }

    /// Returns an equivalent function without block parameters: each edge becomes
    /// `copy tmp_i = arg_i; copy param_i = tmp_i` (parallel assignment), and `cbra` edges
    /// that carry arguments go through a new edge block. Caller owns the result.
    pub fn lowerBlockParams(f: *const Func, gpa: std.mem.Allocator) !Func {
        var g = Func.init(gpa);
        errdefer g.deinit();
        g.name = try g.alloc().dupe(u8, f.name);
        g.params = f.params;
        g.num_vregs = f.num_vregs;
        for (f.blocks.items) |b| _ = try g.addBlock(b.name);
        const Edge = struct {
            fn copies(h: *Func, blk: u32, params: []const VReg, args: []const VReg) !void {
                const base = h.num_vregs;
                h.num_vregs += @intCast(args.len);
                for (args, 0..) |a, i| try h.emit(blk, .{ .copy = .{ .dst = base + @as(VReg, @intCast(i)), .src = a } });
                for (params, 0..) |p, i| try h.emit(blk, .{ .copy = .{ .dst = p, .src = base + @as(VReg, @intCast(i)) } });
            }
            fn split(h: *Func, src: *const Func, target: u32, args: []const VReg) !u32 {
                if (args.len == 0) return target;
                const e = try h.addBlock("edge");
                try copies(h, e, src.blocks.items[target].params, args);
                try h.emit(e, .{ .br = target });
                return e;
            }
        };
        for (f.blocks.items, 0..) |b, bi| {
            const blk: u32 = @intCast(bi);
            for (b.ops.items) |o| switch (o) {
                .bra => |x| {
                    try Edge.copies(&g, blk, f.blocks.items[x.target].params, x.args);
                    try g.emit(blk, .{ .br = x.target });
                },
                .cbra => |x| {
                    const th = try Edge.split(&g, f, x.then, x.then_args);
                    const el = try Edge.split(&g, f, x.@"else", x.else_args);
                    try g.emit(blk, .{ .cbr = .{ .cond = x.cond, .a = x.a, .b = x.b, .then = th, .@"else" = el } });
                },
                else => try g.emit(blk, o),
            };
        }
        return g;
    }

    /// Prints the function in the text format accepted by `parse`.
    pub fn print(f: *const Func, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        try out.print(gpa, "func {s} {d}\n", .{ f.name, f.params });
        for (f.blocks.items) |b| {
            try out.appendSlice(gpa, b.name);
            if (b.params.len != 0) {
                try out.append(gpa, '(');
                for (b.params, 0..) |p, i| try out.print(gpa, "{s}v{d}", .{ if (i == 0) "" else " ", p });
                try out.append(gpa, ')');
            }
            try out.appendSlice(gpa, ":\n");
            for (b.ops.items) |o| {
                try out.appendSlice(gpa, "  ");
                switch (o) {
                    .@"const" => |x| try out.print(gpa, "v{d} = const {d}", .{ x.dst, x.imm }),
                    .copy => |x| try out.print(gpa, "v{d} = copy v{d}", .{ x.dst, x.src }),
                    .arith => |x| try out.print(gpa, "v{d} = {s} v{d} v{d}", .{ x.dst, @tagName(x.op), x.a, x.b }),
                    .load => |x| try out.print(gpa, "v{d} = load{s} v{d} {d}", .{ x.dst, tySuffix(x.ty), x.base, x.disp }),
                    .store => |x| try out.print(gpa, "store{s} v{d} {d} v{d}", .{ tySuffix(x.ty), x.base, x.disp, x.src }),
                    .fconst => |x| if (x.ty == .f32)
                        try out.print(gpa, "v{d} = fconst.f32 {e}", .{ x.dst, @as(f32, @bitCast(@as(u32, @truncate(x.bits)))) })
                    else
                        try out.print(gpa, "v{d} = fconst.f64 {e}", .{ x.dst, @as(f64, @bitCast(x.bits)) }),
                    .typed => |x| try out.print(gpa, "v{d} = {s}.{s} v{d} v{d}", .{ x.dst, @tagName(x.op), @tagName(x.ty), x.a, x.b }),
                    .fcmp => |x| try out.print(gpa, "v{d} = fcmp.{s} {s} v{d} v{d}", .{ x.dst, @tagName(x.ty), @tagName(x.cond), x.a, x.b }),
                    .cvt => |x| try out.print(gpa, "v{d} = cvt.{s}.{s} v{d}", .{ x.dst, @tagName(x.to), @tagName(x.from), x.src }),
                    .splat => |x| try out.print(gpa, "v{d} = splat.{s} v{d}", .{ x.dst, @tagName(x.ty), x.src }),
                    .extract => |x| try out.print(gpa, "v{d} = extract.{s} v{d} {d}", .{ x.dst, @tagName(x.ty), x.src, x.lane }),
                    .call => |x| {
                        try out.print(gpa, "v{d} = call 0x{x}", .{ x.dst, x.addr });
                        for (x.args) |a| try out.print(gpa, " v{d}", .{a});
                    },
                    .br => |tg| try out.print(gpa, "br {s}", .{f.blocks.items[tg].name}),
                    .cbr => |x| try out.print(gpa, "cbr {s} v{d} v{d} {s} {s}", .{ @tagName(x.cond), x.a, x.b, f.blocks.items[x.then].name, f.blocks.items[x.@"else"].name }),
                    .ret => |v| try out.print(gpa, "ret v{d}", .{v}),
                    .bra => |x| {
                        try out.print(gpa, "br {s}", .{f.blocks.items[x.target].name});
                        for (x.args) |a| try out.print(gpa, " v{d}", .{a});
                    },
                    .cbra => |x| {
                        try out.print(gpa, "cbr {s} v{d} v{d} {s}", .{ @tagName(x.cond), x.a, x.b, f.blocks.items[x.then].name });
                        for (x.then_args) |a| try out.print(gpa, ":v{d}", .{a});
                        try out.print(gpa, " {s}", .{f.blocks.items[x.@"else"].name});
                        for (x.else_args) |a| try out.print(gpa, ":v{d}", .{a});
                    },
                }
                try out.append(gpa, '\n');
            }
        }
    }
};

fn tySuffix(t_: Type) []const u8 {
    return switch (t_) {
        .i64 => "",
        inline else => |tt| "." ++ @tagName(tt),
    };
}

/// `mn.suffix` -> (mn, suffix) (suffix empty when there is no '.').
fn splitMn(mn: []const u8) struct { []const u8, []const u8 } {
    const d = std.mem.indexOfScalar(u8, mn, '.') orelse return .{ mn, "" };
    return .{ mn[0..d], mn[d + 1 ..] };
}
fn parseType(s: []const u8) ParseError!Type {
    return std.meta.stringToEnum(Type, s) orelse error.Syntax;
}

pub const ParseError = error{ Syntax, UnknownLabel, DuplicateLabel, OutOfMemory, InvalidCharacter, Overflow, TooManyArgs, BadVReg };

fn parseVReg(tok: ?[]const u8) ParseError!VReg {
    const s = tok orelse return error.Syntax;
    if (s.len < 2 or s[0] != 'v') return error.Syntax;
    const v = try std.fmt.parseInt(VReg, s[1..], 10);
    // maxInt is reserved: `num_vregs = v + 1` must fit.
    return if (v == std.math.maxInt(VReg)) error.Overflow else v;
}
fn parseInt(comptime T: type, tok: ?[]const u8) ParseError!T {
    return std.fmt.parseInt(T, tok orelse return error.Syntax, 0);
}

fn headerName(line: []const u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, line, '(') orelse line.len - 1;
    return std.mem.trim(u8, line[0..end], " \t");
}

/// `name` or `name:vA:vB` (edge with block arguments).
fn edgeTarget(names: *const std.StringHashMapUnmanaged(u32), a: std.mem.Allocator, tok: ?[]const u8, args: *std.ArrayList(VReg)) ParseError!u32 {
    var parts = std.mem.splitScalar(u8, tok orelse return error.Syntax, ':');
    const tg = names.get(parts.next().?) orelse return error.UnknownLabel;
    while (parts.next()) |p| try args.append(a, try parseVReg(p));
    return tg;
}

/// Parses one function from the text format (two passes: labels, then instructions).
pub fn parse(gpa: std.mem.Allocator, src: []const u8) ParseError!Func {
    var f = Func.init(gpa);
    errdefer f.deinit();
    // Pass 1: block names.
    var names: std.StringHashMapUnmanaged(u32) = .empty;
    defer names.deinit(gpa);
    var lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len > 1 and line[line.len - 1] == ':') {
            const name = headerName(line);
            if (names.contains(name)) return error.DuplicateLabel;
            try names.put(gpa, name, try f.addBlock(name));
        }
    }
    var cur: ?u32 = null;
    lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len == 0) continue;
        if (line[line.len - 1] == ':') {
            // Pass 1 skipped one-character lines (a lone ':'): not a block header.
            cur = names.get(headerName(line)) orelse return error.Syntax;
            if (std.mem.indexOfScalar(u8, line, '(')) |lp| {
                const rp = std.mem.lastIndexOfScalar(u8, line, ')') orelse return error.Syntax;
                if (rp < lp) return error.Syntax;
                var ps: std.ArrayList(VReg) = .empty;
                var pit = std.mem.tokenizeAny(u8, line[lp + 1 .. rp], " \t,");
                while (pit.next()) |tok| {
                    const v = try parseVReg(tok);
                    try ps.append(f.alloc(), v);
                    f.num_vregs = @max(f.num_vregs, v + 1);
                }
                f.blocks.items[cur.?].params = ps.items;
            }
            continue;
        }
        var it = std.mem.tokenizeAny(u8, line, " \t,");
        const first = it.next().?;
        if (std.mem.eql(u8, first, "func")) {
            f.name = try f.alloc().dupe(u8, it.next() orelse return error.Syntax);
            f.params = try parseInt(u32, it.next());
            f.num_vregs = @max(f.num_vregs, f.params);
            continue;
        }
        const blk = cur orelse return error.Syntax;
        const label = struct {
            fn get(n: *const std.StringHashMapUnmanaged(u32), tok: ?[]const u8) ParseError!u32 {
                return n.get(tok orelse return error.Syntax) orelse error.UnknownLabel;
            }
        }.get;
        const op: Op = if (std.mem.eql(u8, first, "br")) blk2: {
            const tg = try label(&names, it.next());
            var args: std.ArrayList(VReg) = .empty;
            while (it.next()) |a| try args.append(f.alloc(), try parseVReg(a));
            break :blk2 if (args.items.len == 0) .{ .br = tg } else .{ .bra = .{ .target = tg, .args = args.items } };
        } else if (std.mem.eql(u8, first, "cbr")) blk2: {
            const c = std.meta.stringToEnum(CondCode, it.next() orelse return error.Syntax) orelse return error.Syntax;
            const a = try parseVReg(it.next());
            const b = try parseVReg(it.next());
            var ta: std.ArrayList(VReg) = .empty;
            var ea: std.ArrayList(VReg) = .empty;
            const th = try edgeTarget(&names, f.alloc(), it.next(), &ta);
            const el = try edgeTarget(&names, f.alloc(), it.next(), &ea);
            if (ta.items.len == 0 and ea.items.len == 0) break :blk2 .{ .cbr = .{ .cond = c, .a = a, .b = b, .then = th, .@"else" = el } };
            break :blk2 .{ .cbra = .{ .cond = c, .a = a, .b = b, .then = th, .@"else" = el, .then_args = ta.items, .else_args = ea.items } };
        } else if (std.mem.eql(u8, first, "ret"))
            .{ .ret = try parseVReg(it.next()) }
        else if (std.mem.eql(u8, splitMn(first)[0], "store")) blk2: {
            const sfx = splitMn(first)[1];
            const sty: Type = if (sfx.len == 0) .i64 else try parseType(sfx);
            const base = try parseVReg(it.next());
            const disp = try parseInt(i32, it.next());
            break :blk2 .{ .store = .{ .base = base, .disp = disp, .src = try parseVReg(it.next()), .ty = sty } };
        } else blk2: {
            const dst = try parseVReg(first);
            if (!std.mem.eql(u8, it.next() orelse "", "=")) return error.Syntax;
            const full = it.next() orelse return error.Syntax;
            const mn, const sfx = splitMn(full);
            if (sfx.len != 0) {
                if (std.mem.eql(u8, mn, "load")) {
                    const base = try parseVReg(it.next());
                    break :blk2 .{ .load = .{ .dst = dst, .base = base, .disp = try parseInt(i32, it.next()), .ty = try parseType(sfx) } };
                }
                if (std.mem.eql(u8, mn, "fconst")) {
                    const fty = try parseType(sfx);
                    const v = std.fmt.parseFloat(f64, it.next() orelse return error.Syntax) catch return error.Syntax;
                    const bits: u64 = switch (fty) {
                        .f32 => @as(u32, @bitCast(@as(f32, @floatCast(v)))),
                        .f64 => @bitCast(v),
                        else => return error.Syntax,
                    };
                    break :blk2 .{ .fconst = .{ .dst = dst, .ty = fty, .bits = bits } };
                }
                if (std.mem.eql(u8, mn, "fcmp")) {
                    const c = std.meta.stringToEnum(FCond, it.next() orelse return error.Syntax) orelse return error.Syntax;
                    const a = try parseVReg(it.next());
                    break :blk2 .{ .fcmp = .{ .cond = c, .ty = try parseType(sfx), .dst = dst, .a = a, .b = try parseVReg(it.next()) } };
                }
                if (std.mem.eql(u8, mn, "cvt")) {
                    const to_s, const from_s = splitMn(sfx);
                    break :blk2 .{ .cvt = .{ .to = try parseType(to_s), .from = try parseType(from_s), .dst = dst, .src = try parseVReg(it.next()) } };
                }
                if (std.mem.eql(u8, mn, "splat")) break :blk2 .{ .splat = .{ .ty = try parseType(sfx), .dst = dst, .src = try parseVReg(it.next()) } };
                if (std.mem.eql(u8, mn, "extract")) {
                    const src_v = try parseVReg(it.next());
                    break :blk2 .{ .extract = .{ .ty = try parseType(sfx), .dst = dst, .src = src_v, .lane = try parseInt(u8, it.next()) } };
                }
                const top = std.meta.stringToEnum(TArith, mn) orelse return error.Syntax;
                const a = try parseVReg(it.next());
                break :blk2 .{ .typed = .{ .op = top, .ty = try parseType(sfx), .dst = dst, .a = a, .b = try parseVReg(it.next()) } };
            }
            if (std.mem.eql(u8, mn, "const")) break :blk2 .{ .@"const" = .{ .dst = dst, .imm = try parseInt(i64, it.next()) } };
            if (std.mem.eql(u8, mn, "copy")) break :blk2 .{ .copy = .{ .dst = dst, .src = try parseVReg(it.next()) } };
            if (std.mem.eql(u8, mn, "load")) {
                const base = try parseVReg(it.next());
                break :blk2 .{ .load = .{ .dst = dst, .base = base, .disp = try parseInt(i32, it.next()) } };
            }
            if (std.mem.eql(u8, mn, "call")) {
                const addr = try parseInt(usize, it.next());
                var args: std.ArrayList(VReg) = .empty;
                while (it.next()) |a| try args.append(f.alloc(), try parseVReg(a));
                break :blk2 .{ .call = .{ .dst = dst, .addr = addr, .args = args.items } };
            }
            const ar = std.meta.stringToEnum(Arith, mn) orelse return error.Syntax;
            const a = try parseVReg(it.next());
            break :blk2 .{ .arith = .{ .op = ar, .dst = dst, .a = a, .b = try parseVReg(it.next()) } };
        };
        if (it.next() != null) return error.Syntax;
        try f.emit(blk, op);
    }
    return f;
}

/// Reference interpreter. `mem` backs load/store (vA is an address inside it). Calls go to
/// real function pointers. Returns error.StepLimit after `max_steps` instructions.
pub fn interpret(f: *const Func, args: []const i64, max_steps: usize) !i64 {
    const regs = try std.heap.page_allocator.alloc(Val, @max(f.num_vregs, 1));
    defer std.heap.page_allocator.free(regs);
    @memset(regs, 0);
    for (args, 0..) |a, i| regs[i] = fromI(a);
    var b: u32 = 0;
    var steps: usize = 0;
    while (true) {
        for (f.blocks.items[b].ops.items) |o| {
            steps += 1;
            if (steps > max_steps) return error.StepLimit;
            switch (o) {
                .@"const" => |x| regs[x.dst] = fromI(x.imm),
                .copy => |x| regs[x.dst] = regs[x.src],
                .arith => |x| regs[x.dst] = fromI(arith(x.op, toI(regs[x.a]), toI(regs[x.b]))),
                .load => |x| {
                    const p: [*]const u8 = @ptrFromInt(@as(usize, @truncate(@as(u64, @bitCast(toI(regs[x.base]) +% x.disp)))));
                    var v: Val = 0;
                    @memcpy(std.mem.asBytes(&v)[0..x.ty.size()], p[0..x.ty.size()]);
                    regs[x.dst] = v;
                },
                .store => |x| {
                    const p: [*]u8 = @ptrFromInt(@as(usize, @truncate(@as(u64, @bitCast(toI(regs[x.base]) +% x.disp)))));
                    @memcpy(p[0..x.ty.size()], std.mem.asBytes(&regs[x.src])[0..x.ty.size()]);
                },
                .call => |x| {
                    // Up to `max_args` (what `verify` accepts); unused trailing args are 0.
                    const F = fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
                    var a: std.meta.ArgsTuple(F) = undefined;
                    inline for (0..max_args) |i| a[i] = if (i < x.args.len) toI(regs[x.args[i]]) else 0;
                    regs[x.dst] = fromI(@call(.auto, @as(*const F, @ptrFromInt(x.addr)), a));
                },
                .fconst => |x| regs[x.dst] = x.bits,
                .typed => |x| regs[x.dst] = typedArith(x.op, x.ty, regs[x.a], regs[x.b]),
                .fcmp => |x| regs[x.dst] = @intFromBool(if (x.ty == .f32)
                    fcond(x.cond, getF32(regs[x.a]), getF32(regs[x.b]))
                else
                    fcond(x.cond, getF64(regs[x.a]), getF64(regs[x.b]))),
                .cvt => |x| regs[x.dst] = convert(x.to, x.from, regs[x.src]),
                .splat => |x| {
                    var v: Val = 0;
                    for (0..x.ty.lanes()) |l| v = setLane(x.ty, v, @intCast(l), regs[x.src]);
                    regs[x.dst] = v;
                },
                .extract => |x| regs[x.dst] = getLane(x.ty, regs[x.src], x.lane),
                .br => |tg| {
                    b = tg;
                    break;
                },
                .cbr => |x| {
                    b = if (cond(x.cond, toI(regs[x.a]), toI(regs[x.b]))) x.then else x.@"else";
                    break;
                },
                .bra => |x| {
                    assignParams(regs, f.blocks.items[x.target].params, x.args);
                    b = x.target;
                    break;
                },
                .cbra => |x| {
                    const taken = cond(x.cond, toI(regs[x.a]), toI(regs[x.b]));
                    b = if (taken) x.then else x.@"else";
                    assignParams(regs, f.blocks.items[b].params, if (taken) x.then_args else x.else_args);
                    break;
                },
                .ret => |v| return toI(regs[v]),
            }
        }
    }
}

fn fromI(v: i64) Val {
    return @as(u64, @bitCast(v));
}
fn toI(v: Val) i64 {
    return @bitCast(@as(u64, @truncate(v)));
}
pub fn getF32(v: Val) f32 {
    return @bitCast(@as(u32, @truncate(v)));
}
pub fn getF64(v: Val) f64 {
    return @bitCast(@as(u64, @truncate(v)));
}
pub fn fromF32(x: f32) Val {
    return @as(u32, @bitCast(x));
}
pub fn fromF64(x: f64) Val {
    return @as(u64, @bitCast(x));
}
fn fcond(c: FCond, a: anytype, b: @TypeOf(a)) bool {
    return switch (c) {
        .eq => a == b,
        .ne => a != b,
        .lt => a < b,
        .le => a <= b,
        .gt => a > b,
        .ge => a >= b,
    };
}
/// Lane `l` of vector type `t` as a scalar value (integer lanes sign-extended to i64).
pub fn getLane(tp: Type, v: Val, l: u8) Val {
    return switch (tp) {
        .v4i32 => fromI(@as(i32, @bitCast(@as(u32, @truncate(v >> (32 * @as(u7, @intCast(l)))))))),
        .v4f32 => @as(u32, @truncate(v >> (32 * @as(u7, @intCast(l))))),
        .v2i64, .v2f64 => @as(u64, @truncate(v >> (64 * @as(u7, @intCast(l))))),
        else => v,
    };
}
fn setLane(tp: Type, v: Val, l: u8, x: Val) Val {
    const w: u7 = if (tp == .v4i32 or tp == .v4f32) 32 else 64;
    const sh: u7 = w * @as(u7, @intCast(l));
    const mask: Val = (@as(Val, 1) << w) - 1;
    return (v & ~(mask << sh)) | ((x & mask) << sh);
}
fn scalarArith(op: TArith, tp: Type, a: Val, b: Val) Val {
    switch (tp) {
        .f32, .f64 => {
            const F = struct {
                fn run(comptime T: type, o: TArith, x: T, y: T) T {
                    return switch (o) {
                        .add => x + y,
                        .sub => x - y,
                        .mul => x * y,
                        .div => x / y,
                        // x86 minss/maxss semantics: the second operand when unordered or equal.
                        .min => if (x < y) x else y,
                        .max => if (x > y) x else y,
                        else => unreachable,
                    };
                }
            };
            return if (tp == .f32) fromF32(F.run(f32, op, getF32(a), getF32(b))) else fromF64(F.run(f64, op, getF64(a), getF64(b)));
        },
        else => {
            const x = toI(a);
            const y = toI(b);
            return fromI(switch (op) {
                .add => x +% y,
                .sub => x -% y,
                .mul => x *% y,
                .@"and" => x & y,
                .@"or" => x | y,
                .xor => x ^ y,
                else => unreachable,
            });
        },
    }
}
pub fn typedArith(op: TArith, tp: Type, a: Val, b: Val) Val {
    if (!tp.isVector()) return scalarArith(op, tp, a, b);
    var r: Val = 0;
    for (0..tp.lanes()) |l| r = setLane(tp, r, @intCast(l), scalarArith(op, tp.lane(), getLane(tp, a, @intCast(l)), getLane(tp, b, @intCast(l))));
    return r;
}
fn convert(to: Type, from: Type, v: Val) Val {
    return switch (to) {
        .i64 => fromI(if (from == .f32) floatToI(getF32(v)) else floatToI(getF64(v))),
        .f32 => if (from == .i64) fromF32(@floatFromInt(toI(v))) else fromF32(@floatCast(getF64(v))),
        .f64 => if (from == .i64) fromF64(@floatFromInt(toI(v))) else fromF64(getF32(v)),
        else => unreachable,
    };
}
/// Truncating float -> i64 (x86 `cvttsd2si`: NaN / out of range give minInt(i64)).
fn floatToI(x: anytype) i64 {
    if (std.math.isNan(x) or x >= 9.223372036854775807e18 or x < -9.223372036854775808e18) return std.math.minInt(i64);
    return @intFromFloat(x);
}

fn assignParams(regs: []Val, params: []const VReg, args: []const VReg) void {
    var tmp: [max_args]Val = undefined;
    for (args, 0..) |a, i| tmp[i] = regs[a];
    for (params, 0..) |p, i| regs[p] = tmp[i];
}

pub fn arith(op: Arith, a: i64, b: i64) i64 {
    return switch (op) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .@"and" => a & b,
        .@"or" => a | b,
        .xor => a ^ b,
        .shl => @bitCast(@as(u64, @bitCast(a)) << @intCast(b & 63)),
        .shr => @bitCast(@as(u64, @bitCast(a)) >> @intCast(b & 63)),
        .sar => a >> @intCast(b & 63),
        // Undefined cases (b == 0, minInt / -1) return 0 here; callers must avoid them.
        .sdiv => if (b == 0 or (a == std.math.minInt(i64) and b == -1)) 0 else @divTrunc(a, b),
        .srem => if (b == 0 or (a == std.math.minInt(i64) and b == -1)) 0 else @rem(a, b),
    };
}
pub fn cond(c: CondCode, a: i64, b: i64) bool {
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

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;

pub const fib_src =
    \\func fib 1
    \\entry:
    \\  v1 = const 0      # a
    \\  v2 = const 1      # b
    \\  v3 = const 0      # i
    \\  v5 = const 1
    \\  br loop
    \\loop:
    \\  cbr lt v3 v0 body done
    \\body:
    \\  v4 = add v1 v2
    \\  v1 = copy v2
    \\  v2 = copy v4
    \\  v3 = add v3 v5
    \\  br loop
    \\done:
    \\  ret v1
;

test "ir: parse, verify, print round-trip, interpret" {
    var f = try parse(t.allocator, fib_src);
    defer f.deinit();
    try f.verify();
    try t.expectEqual(@as(i64, 55), try interpret(&f, &.{10}, 10_000));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try f.print(t.allocator, &out);
    var g = try parse(t.allocator, out.items);
    defer g.deinit();
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(t.allocator);
    try g.print(t.allocator, &out2);
    try t.expectEqualStrings(out.items, out2.items);
}

test "ir: verifier and parser reject bad input" {
    try t.expectError(error.UnknownLabel, parse(t.allocator, "func f 0\na:\n  br nowhere\n"));
    try t.expectError(error.Syntax, parse(t.allocator, "func f 0\na:\n  v0 = frob v1 v2\n"));
    try t.expectError(error.DuplicateLabel, parse(t.allocator, "func f 0\na:\na:\n"));
    var f = try parse(t.allocator, "func f 1\na:\n  v1 = add v0 v0\n");
    defer f.deinit();
    try t.expectError(error.MissingTerminator, f.verify());
    var g = Func.init(t.allocator);
    defer g.deinit();
    const b = try g.addBlock("a");
    try g.emit(b, .{ .ret = 0 });
    try g.emit(b, .{ .ret = 0 });
    try t.expectError(error.TerminatorNotLast, g.verify());
}

pub const fib_ssa_src =
    \\func fib 1
    \\entry:
    \\  v1 = const 0
    \\  v2 = const 1
    \\  v3 = const 0
    \\  v4 = const 1
    \\  br loop v1 v2 v3
    \\loop(v5 v6 v7):
    \\  cbr lt v7 v0 body done:v5
    \\body:
    \\  v8 = add v5 v6
    \\  v9 = add v7 v4
    \\  br loop v6 v8 v9
    \\done(v10):
    \\  ret v10
;

test "ir: block parameters - parse, SSA verify, print round-trip, interpret, lower" {
    var f = try parse(t.allocator, fib_ssa_src);
    defer f.deinit();
    try f.verifySsa(t.allocator);
    try t.expect(f.hasBlockParams());
    for ([_]i64{ 0, 1, 2, 10, 50 }) |n| {
        var p = try parse(t.allocator, fib_src);
        defer p.deinit();
        const want = try interpret(&p, &.{n}, 1 << 20);
        try t.expectEqual(want, try interpret(&f, &.{n}, 1 << 20));
        var g = try f.lowerBlockParams(t.allocator);
        defer g.deinit();
        try g.verify();
        try t.expect(!g.hasBlockParams());
        try t.expectEqual(want, try interpret(&g, &.{n}, 1 << 20));
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try f.print(t.allocator, &out);
    try t.expect(std.mem.indexOf(u8, out.items, "loop(v5 v6 v7):") != null);
    var g = try parse(t.allocator, out.items);
    defer g.deinit();
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(t.allocator);
    try g.print(t.allocator, &out2);
    try t.expectEqualStrings(out.items, out2.items);
}

test "ir: block parameters - parallel swap and verifier errors" {
    // b(v1 v2) receives (v2 v1): a swap that sequential copies would get wrong.
    const src =
        \\func f 2
        \\e:
        \\  br s v1 v0
        \\s(v2 v3):
        \\  br b v3 v2
        \\b(v4 v5):
        \\  v6 = const 10
        \\  v7 = mul v4 v6
        \\  v8 = add v7 v5
        \\  ret v8
    ;
    var f = try parse(t.allocator, src);
    defer f.deinit();
    try f.verifySsa(t.allocator);
    try t.expectEqual(@as(i64, 12), try interpret(&f, &.{ 1, 2 }, 100));
    var g = try f.lowerBlockParams(t.allocator);
    defer g.deinit();
    try t.expectEqual(@as(i64, 12), try interpret(&g, &.{ 1, 2 }, 100));

    var bad = try parse(t.allocator, "func f 1\na:\n  br b v0\nb(v1 v2):\n  ret v1\n");
    defer bad.deinit();
    try t.expectError(error.ArgCountMismatch, bad.verify());
    var nossa = try parse(t.allocator, "func f 1\na:\n  v1 = const 1\n  v1 = const 2\n  ret v1\n");
    defer nossa.deinit();
    try nossa.verify();
    try t.expectError(error.MultipleDefs, nossa.verifySsa(t.allocator));
}

test "ir: typed values - parse, type check, print round-trip, interpret" {
    const src =
        \\func k 2
        \\e:
        \\  v2 = cvt.f64.i64 v0
        \\  v3 = fconst.f64 0.5
        \\  v4 = mul.f64 v2 v3
        \\  v5 = cvt.f32.f64 v4
        \\  v6 = splat.v4f32 v5
        \\  v7 = add.v4f32 v6 v6
        \\  v8 = extract.v4f32 v7 3
        \\  v9 = cvt.i64.f32 v8
        \\  v10 = splat.v4i32 v1
        \\  v11 = mul.v4i32 v10 v10
        \\  v12 = extract.v4i32 v11 2
        \\  v13 = add v9 v12
        \\  v14 = fcmp.f64 lt v2 v4
        \\  v15 = add v13 v14
        \\  ret v15
    ;
    var f = try parse(t.allocator, src);
    defer f.deinit();
    try f.verify();
    try t.expect(!f.isIntOnly());
    const ty = try f.typeCheck(t.allocator);
    defer t.allocator.free(ty);
    try t.expectEqual(Type.f64, ty[4]);
    try t.expectEqual(Type.v4f32, ty[7]);
    try t.expectEqual(Type.f32, ty[8]);
    try t.expectEqual(Type.i64, ty[12]);
    // v0 = 6: 6*0.5 = 3 -> f32 3 -> splat+add 6 -> i64 6; v1 = 5: 25; lt(6.0, 3.0) = 0
    try t.expectEqual(@as(i64, 31), try interpret(&f, &.{ 6, 5 }, 100));
    // negative: lt(-4.0, -2.0) = 1; -2 -> -4 ; v1 = -3 -> 9 => -4 + 9 + 1
    try t.expectEqual(@as(i64, 6), try interpret(&f, &.{ -4, -3 }, 100));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    try f.print(t.allocator, &out);
    var g = try parse(t.allocator, out.items);
    defer g.deinit();
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(t.allocator);
    try g.print(t.allocator, &out2);
    try t.expectEqualStrings(out.items, out2.items);
    try t.expectEqual(@as(i64, 31), try interpret(&g, &.{ 6, 5 }, 100));
}

test "ir: typed memory and type errors" {
    var mem: [4]f64 = .{ 1.5, 2.25, 0, 0 };
    const p: i64 = @intCast(@intFromPtr(&mem));
    const src =
        \\func m 1
        \\e:
        \\  v1 = load.v2f64 v0 0
        \\  v2 = add.v2f64 v1 v1
        \\  store.v2f64 v0 16 v2
        \\  v3 = load.f64 v0 24
        \\  v4 = cvt.i64.f64 v3
        \\  ret v4
    ;
    var f = try parse(t.allocator, src);
    defer f.deinit();
    const ty = try f.typeCheck(t.allocator);
    t.allocator.free(ty);
    try t.expectEqual(@as(i64, 4), try interpret(&f, &.{p}, 100));
    try t.expectEqual(@as(f64, 3.0), mem[2]);
    const bad = [_][]const u8{
        "func b 1\ne:\n  v1 = fconst.f64 1\n  v2 = add v1 v0\n  ret v2\n", // i64 add on f64
        "func b 1\ne:\n  v1 = div.v4i32 v0 v0\n  ret v0\n", // div on int vector (and operand types)
        "func b 1\ne:\n  v1 = fconst.f32 1\n  ret v1\n", // ret needs i64
        "func b 1\ne:\n  v1 = splat.v2f64 v0\n  ret v0\n", // lane type f64, got i64
        "func b 1\ne:\n  v1 = fconst.f64 1\n  v2 = extract.v2f64 v1 0\n  ret v0\n",
        "func b 1\ne:\n  v1 = fconst.f64 1\n  v1 = fconst.f32 1\n  ret v0\n", // conflicting defs
    };
    for (bad) |b| {
        var g = try parse(t.allocator, b);
        defer g.deinit();
        try t.expectError(error.TypeMismatch, g.typeCheck(t.allocator));
    }
}

test "ir: int-only check" {
    var f = try parse(t.allocator, fib_src);
    defer f.deinit();
    try t.expect(f.isIntOnly());
    var g = try parse(t.allocator, "func b 0\ne:\n  v0 = fconst.f64 1\n  v1 = cvt.i64.f64 v0\n  ret v1\n");
    defer g.deinit();
    try t.expect(!g.isIntOnly());
    try t.expectEqual(@as(i64, 1), try interpret(&g, &.{}, 10));
}

test "ir: malformed input is an error, not a panic" {
    const bad = [_][]const u8{
        "func f 0\n:\n",
        "func f 0\na)(:\n  ret v0\n",
        "func f 0\na:\n  v4294967295 = const 1\n  ret v0\n",
        "func f 1\na:\n  v1 = call 0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0\n  ret v1\n",
        "func f 1\na:\n  br a v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0 v0\n",
    };
    for (bad) |src| {
        if (parse(std.testing.allocator, src)) |f_| {
            var f = f_;
            f.deinit();
            return error.TestUnexpectedResult;
        } else |_| {}
    }
}

fn sum7(a: i64, b: i64, c: i64, d: i64, e: i64, g: i64, h: i64) callconv(.c) i64 {
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * g + 7 * h;
}

test "ir: interpreter calls with more than 6 arguments" {
    var f = Func.init(std.testing.allocator);
    defer f.deinit();
    f.params = 1;
    f.num_vregs = 1;
    const e = try f.addBlock("e");
    const args = try f.alloc().dupe(VReg, &.{ 0, 0, 0, 0, 0, 0, 0 });
    try f.emit(e, .{ .call = .{ .dst = 1, .addr = @intFromPtr(&sum7), .args = args } });
    try f.emit(e, .{ .ret = 1 });
    try f.verify();
    try std.testing.expectEqual(@as(i64, 28), try interpret(&f, &.{1}, 100));
}
