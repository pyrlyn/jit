//! Host CPU feature detection for x86-64 (CPUID). Returns null on other hosts.
//! Covers the GP/SSE/AVX bits relevant to code generation; extend `detect` as forms are added.
const std = @import("std");
const builtin = @import("builtin");
const isa = @import("isa.zig");

fn cpuid(leaf: u32, sub: u32) [4]u32 {
    var a: u32 = undefined;
    var b: u32 = undefined;
    var c: u32 = undefined;
    var d: u32 = undefined;
    asm volatile ("cpuid"
        : [a] "={eax}" (a),
          [b] "={ebx}" (b),
          [c] "={ecx}" (c),
          [d] "={edx}" (d),
        : [leaf] "{eax}" (leaf),
          [sub] "{ecx}" (sub),
    );
    return .{ a, b, c, d };
}

fn detect() isa.Features {
    var f: isa.Features = .initEmpty();
    const max = cpuid(0, 0)[0];
    const l1 = cpuid(1, 0);
    const bit = struct {
        fn on(v: u32, n: u5) bool {
            return (v >> n) & 1 != 0;
        }
    }.on;
    f.setPresent(.SSE, bit(l1[3], 25));
    f.setPresent(.SSE2, bit(l1[3], 26));
    f.setPresent(.SSE3, bit(l1[2], 0));
    f.setPresent(.SSSE3, bit(l1[2], 9));
    f.setPresent(.FMA, bit(l1[2], 12));
    f.setPresent(.SSE4_1, bit(l1[2], 19));
    f.setPresent(.SSE4_2, bit(l1[2], 20));
    f.setPresent(.MOVBE, bit(l1[2], 22));
    f.setPresent(.POPCNT, bit(l1[2], 23));
    f.setPresent(.F16C, bit(l1[2], 29));
    // AVX also needs OS support (OSXSAVE + XCR0 YMM state); checked conservatively.
    const avx_os = bit(l1[2], 27) and bit(l1[2], 28) and (xgetbv0() & 0x6) == 0x6;
    f.setPresent(.AVX, avx_os);
    if (max >= 7) {
        const l7 = cpuid(7, 0);
        f.setPresent(.BMI, bit(l7[1], 3));
        f.setPresent(.AVX2, avx_os and bit(l7[1], 5));
        f.setPresent(.BMI2, bit(l7[1], 8));
        const zmm_os = avx_os and (xgetbv0() & 0xE6) == 0xE6;
        f.setPresent(.AVX512_F, zmm_os and bit(l7[1], 16));
        f.setPresent(.AVX512_DQ, zmm_os and bit(l7[1], 17));
        f.setPresent(.AVX512_BW, zmm_os and bit(l7[1], 30));
    }
    const ext = cpuid(0x80000000, 0)[0];
    if (ext >= 0x80000001) f.setPresent(.LZCNT, bit(cpuid(0x80000001, 0)[2], 5));
    return f;
}

fn xgetbv0() u32 {
    return asm volatile ("xgetbv"
        : [ret] "={eax}" (-> u32),
        : [idx] "{ecx}" (@as(u32, 0)),
        : .{ .edx = true });
}

var cached: ?isa.Features = null;

/// Features of the CPU running this process, or null if the host is not x86-64.
pub fn host() ?isa.Features {
    if (builtin.cpu.arch != .x86_64) return null;
    if (cached == null) cached = detect();
    return cached;
}
