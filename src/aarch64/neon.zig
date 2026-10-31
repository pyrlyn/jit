//! AArch64 NEON (Advanced SIMD) basics on v0..v31, 128-bit arrangements. Byte-checked against
//! LLVM in `src/simd_tests.zig` (scripts/gen_simd_tests.py). NEON is baseline on AArch64.
const std = @import("std");
const core = @import("../core/code_buffer.zig");
const CodeBuffer = core.CodeBuffer;
const X = @import("assembler.zig").X;

pub const Error = core.Error || error{ InvalidOperand, InvalidImmediate };
pub const V = u5;
/// 128-bit arrangement: 16 bytes, 8 halves, 4 singles, 2 doubles.
pub const Arr = enum(u2) { b16 = 0, h8 = 1, s4 = 2, d2 = 3 };

fn put(cb: *CodeBuffer, w: u32) Error!void {
    try cb.emitBytes(std.mem.asBytes(&std.mem.nativeToLittle(u32, w)));
}
fn rrr(base: u32, rd: V, rn: V, rm: V) u32 {
    return base | @as(u32, rm) << 16 | @as(u32, rn) << 5 | rd;
}
fn fsz(a: Arr) Error!u32 {
    return switch (a) {
        .s4 => 0,
        .d2 => 1 << 22,
        else => error.InvalidOperand,
    };
}
fn xid(r: X) u32 {
    return @backingInt(r);
}

pub fn add(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x4e208400 | @as(u32, @backingInt(a)) << 22, rd, rn, rm));
}
pub fn sub(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x6e208400 | @as(u32, @backingInt(a)) << 22, rd, rn, rm));
}
pub fn mul(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    if (a == .d2) return error.InvalidOperand;
    return put(cb, rrr(0x4e209c00 | @as(u32, @backingInt(a)) << 22, rd, rn, rm));
}
pub fn fadd(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x4e20d400 | try fsz(a), rd, rn, rm));
}
pub fn fsub(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x4ea0d400 | try fsz(a), rd, rn, rm));
}
pub fn fmul(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x6e20dc00 | try fsz(a), rd, rn, rm));
}
/// rd += rn * rm (fused).
pub fn fmla(cb: *CodeBuffer, a: Arr, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x4e20cc00 | try fsz(a), rd, rn, rm));
}
pub fn andv(cb: *CodeBuffer, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x4e201c00, rd, rn, rm));
}
pub fn orr(cb: *CodeBuffer, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x4ea01c00, rd, rn, rm));
}
pub fn eor(cb: *CodeBuffer, rd: V, rn: V, rm: V) Error!void {
    return put(cb, rrr(0x6e201c00, rd, rn, rm));
}
pub fn movV(cb: *CodeBuffer, rd: V, rn: V) Error!void {
    return orr(cb, rd, rn, rn);
}
/// ldr qT, [xN|sp, #off] (off multiple of 16, < 65536).
pub fn ldrQ(cb: *CodeBuffer, rt: V, rn: X, off: u32) Error!void {
    if (off % 16 != 0 or off / 16 > 4095) return error.InvalidImmediate;
    return put(cb, 0x3dc00000 | (off / 16) << 10 | xid(rn) << 5 | rt);
}
pub fn strQ(cb: *CodeBuffer, rt: V, rn: X, off: u32) Error!void {
    if (off % 16 != 0 or off / 16 > 4095) return error.InvalidImmediate;
    return put(cb, 0x3d800000 | (off / 16) << 10 | xid(rn) << 5 | rt);
}
/// dup vD.4s, wN
pub fn dupS(cb: *CodeBuffer, rd: V, rn: X) Error!void {
    return put(cb, 0x4e040c00 | xid(rn) << 5 | rd);
}
/// mov wD, vN.s[lane]  (umov)
pub fn umovS(cb: *CodeBuffer, rd: X, rn: V, lane: u2) Error!void {
    return put(cb, 0x0e003c00 | (@as(u32, lane) << 3 | 4) << 16 | @as(u32, rn) << 5 | xid(rd));
}
