//! Host CPU feature detection for AArch64. macOS: `sysctlbyname("hw.optional.arm.FEAT_*")`.
//! Returns null on other hosts or OSes (extend for Linux via HWCAP when needed).
const std = @import("std");
const builtin = @import("builtin");
const isa = @import("isa.zig");

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*anyopaque, newlen: usize) c_int;

fn has(name: [*:0]const u8) bool {
    var v: i32 = 0;
    var len: usize = @sizeOf(i32);
    return sysctlbyname(name, &v, &len, null, 0) == 0 and v != 0;
}

fn detect() isa.Features {
    var f: isa.Features = .empty;
    f.insert(.ASIMD); // mandatory on Apple silicon
    const map = .{
        .{ .LSE, "hw.optional.arm.FEAT_LSE" },       .{ .CRC32, "hw.optional.armv8_crc32" },
        .{ .AES, "hw.optional.arm.FEAT_AES" },       .{ .SHA1, "hw.optional.arm.FEAT_SHA1" },
        .{ .SHA256, "hw.optional.arm.FEAT_SHA256" }, .{ .SHA3, "hw.optional.arm.FEAT_SHA3" },
        .{ .SHA512, "hw.optional.arm.FEAT_SHA512" }, .{ .DOTPROD, "hw.optional.arm.FEAT_DotProd" },
        .{ .FP16, "hw.optional.arm.FEAT_FP16" },     .{ .FHM, "hw.optional.arm.FEAT_FHM" },
        .{ .RDM, "hw.optional.arm.FEAT_RDM" },       .{ .FCMA, "hw.optional.arm.FEAT_FCMA" },
        .{ .JSCVT, "hw.optional.arm.FEAT_JSCVT" },   .{ .LRCPC, "hw.optional.arm.FEAT_LRCPC" },
        .{ .LRCPC2, "hw.optional.arm.FEAT_LRCPC2" }, .{ .BF16, "hw.optional.arm.FEAT_BF16" },
        .{ .I8MM, "hw.optional.arm.FEAT_I8MM" },     .{ .SB, "hw.optional.arm.FEAT_SB" },
        .{ .BTI, "hw.optional.arm.FEAT_BTI" },       .{ .PAUTH, "hw.optional.arm.FEAT_PAuth" },
        .{ .FLAGM, "hw.optional.arm.FEAT_FlagM" },   .{ .FLAGM2, "hw.optional.arm.FEAT_FlagM2" },
        .{ .SME, "hw.optional.arm.FEAT_SME" },
    };
    inline for (map) |m| f.setPresent(m[0], has(m[1]));
    return f;
}

var cached: ?isa.Features = null;

/// Features of the CPU running this process, or null if not AArch64 macOS.
pub fn host() ?isa.Features {
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .macos) return null;
    if (cached == null) cached = detect();
    return cached;
}
