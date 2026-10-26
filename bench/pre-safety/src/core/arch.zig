//! Target description (AsmJit: Environment / Arch). Host and target are independent:
//! you can generate x86-64 code on an AArch64 host and vice versa; only execution
//! through JitRuntime requires target == host.
const builtin = @import("builtin");

pub const Arch = enum { x86_64, aarch64, x86 };

pub const Environment = struct {
    arch: Arch,
    pub fn host() Environment {
        return .{ .arch = switch (builtin.cpu.arch) {
            .x86_64 => .x86_64,
            .aarch64 => .aarch64,
            .x86 => .x86,
            else => @compileError("unsupported host arch"),
        } };
    }
};
