//! Checks that the LLVM-derived validation tables (src/x86_64/isa.zig, src/aarch64/isa.zig) carry
//! the same encodings and feature names as the previous AsmJit-derived tables (commit 88245c9),
//! apart from the reviewed changes listed in the tests below.
//!
//! The expected values are the encoding facts of the previous tables (no AsmJit text), frozen here.
const std = @import("std");
const t = std.testing;
const opts = @import("pyrlyn_jit_options");
const x_isa = @import("x86_64/isa.zig");
const a_isa = @import("aarch64/isa.zig");

const X = struct { opcode: []const u8, plus: x_isa.Plus, modrm: bool, digit: ?u3, imm: x_isa.Imm, features: []const x_isa.Feature };
const A = struct { mask: u32, value: u32, rd_sp: bool, rn_sp: bool, imm_bits: u8, imm_scale: u8, features: []const a_isa.Feature };

const x_prev = struct {
    pub const mov_rr = [_]X{.{ .opcode = &.{0x89}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const mov_mr = [_]X{.{ .opcode = &.{0x89}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const mov_rm = [_]X{.{ .opcode = &.{0x8B}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const mov_mi = [_]X{.{ .opcode = &.{0xC7}, .plus = .none, .modrm = true, .digit = 0, .imm = .imm32, .features = &.{} }};
    pub const lea = [_]X{.{ .opcode = &.{0x8D}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const mov_ri = [_]X{ .{ .opcode = &.{0xB8}, .plus = .reg, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0xC7}, .plus = .none, .modrm = true, .digit = 0, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0xB8}, .plus = .reg, .modrm = false, .digit = null, .imm = .imm64, .features = &.{} } };
    pub const movabs = [_]X{.{ .opcode = &.{0xB8}, .plus = .reg, .modrm = false, .digit = null, .imm = .imm64, .features = &.{} }};
    pub const test_rr = [_]X{.{ .opcode = &.{0x85}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const imul_rr = [_]X{.{ .opcode = &.{ 0x0F, 0xAF }, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const imul_rm = [_]X{.{ .opcode = &.{ 0x0F, 0xAF }, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }};
    pub const imul_rri = [_]X{ .{ .opcode = &.{0x6B}, .plus = .none, .modrm = true, .digit = null, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x69}, .plus = .none, .modrm = true, .digit = null, .imm = .imm32, .features = &.{} } };
    pub const push = [_]X{.{ .opcode = &.{0x50}, .plus = .reg, .modrm = false, .digit = null, .imm = .none, .features = &.{} }};
    pub const pop = [_]X{.{ .opcode = &.{0x58}, .plus = .reg, .modrm = false, .digit = null, .imm = .none, .features = &.{} }};
    pub const ret = [_]X{.{ .opcode = &.{0xC3}, .plus = .none, .modrm = false, .digit = null, .imm = .none, .features = &.{} }};
    pub const nop = [_]X{.{ .opcode = &.{0x90}, .plus = .none, .modrm = false, .digit = null, .imm = .none, .features = &.{} }};
    pub const call_r = [_]X{.{ .opcode = &.{0xFF}, .plus = .none, .modrm = true, .digit = 2, .imm = .none, .features = &.{} }};
    pub const jmp_r = [_]X{.{ .opcode = &.{0xFF}, .plus = .none, .modrm = true, .digit = 4, .imm = .none, .features = &.{} }};
    pub const jmp = [_]X{ .{ .opcode = &.{0xEB}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{0xE9}, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } };
    pub const call = [_]X{.{ .opcode = &.{0xE8}, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} }};
    pub const alu_rr = [8][]const X{
        &.{.{ .opcode = &.{0x01}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x09}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x11}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x19}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x21}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x29}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x31}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x39}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
    };
    pub const alu_mr = [8][]const X{
        &.{.{ .opcode = &.{0x01}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x09}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x11}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x19}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x21}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x29}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x31}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x39}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
    };
    pub const alu_rm = [8][]const X{
        &.{.{ .opcode = &.{0x03}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x0B}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x13}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x1B}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x23}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x2B}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x33}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
        &.{.{ .opcode = &.{0x3B}, .plus = .none, .modrm = true, .digit = null, .imm = .none, .features = &.{} }},
    };
    pub const alu_ri = [8][]const X{
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 0, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 0, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x05}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 1, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 1, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x0D}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 2, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 2, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x15}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 3, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 3, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x1D}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 4, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 4, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x25}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 5, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 5, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x2D}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 6, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 6, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x35}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x83}, .plus = .none, .modrm = true, .digit = 7, .imm = .imm8, .features = &.{} }, .{ .opcode = &.{0x81}, .plus = .none, .modrm = true, .digit = 7, .imm = .imm32, .features = &.{} }, .{ .opcode = &.{0x3D}, .plus = .none, .modrm = false, .digit = null, .imm = .imm32, .features = &.{} } },
    };
    pub const jcc = [16][]const X{
        &.{ .{ .opcode = &.{0x70}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x80 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x71}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x81 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x72}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x82 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x73}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x83 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x74}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x84 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x75}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x85 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x76}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x86 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x77}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x87 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x78}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x88 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x79}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x89 }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x7A}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x8A }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x7B}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x8B }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x7C}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x8C }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x7D}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x8D }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x7E}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x8E }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
        &.{ .{ .opcode = &.{0x7F}, .plus = .none, .modrm = false, .digit = null, .imm = .rel8, .features = &.{} }, .{ .opcode = &.{ 0x0F, 0x8F }, .plus = .none, .modrm = false, .digit = null, .imm = .rel32, .features = &.{} } },
    };
};

const a_prev = struct {
    pub const movz: A = .{ .mask = 0xFF800000, .value = 0xD2800000, .rd_sp = false, .rn_sp = false, .imm_bits = 16, .imm_scale = 1, .features = &.{} };
    pub const movk: A = .{ .mask = 0xFF800000, .value = 0xF2800000, .rd_sp = false, .rn_sp = false, .imm_bits = 16, .imm_scale = 1, .features = &.{} };
    pub const mov: A = .{ .mask = 0xFFE0FFE0, .value = 0xAA0003E0, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const add: A = .{ .mask = 0xFF200000, .value = 0x8B000000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const sub: A = .{ .mask = 0xFF200000, .value = 0xCB000000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const mul: A = .{ .mask = 0xFFE0FC00, .value = 0x9B007C00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const add_imm: A = .{ .mask = 0xFF800000, .value = 0x91000000, .rd_sp = true, .rn_sp = true, .imm_bits = 12, .imm_scale = 1, .features = &.{} };
    pub const sub_imm: A = .{ .mask = 0xFF800000, .value = 0xD1000000, .rd_sp = true, .rn_sp = true, .imm_bits = 12, .imm_scale = 1, .features = &.{} };
    pub const cmp: A = .{ .mask = 0xFF20001F, .value = 0xEB00001F, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const cmp_imm: A = .{ .mask = 0xFF80001F, .value = 0xF100001F, .rd_sp = false, .rn_sp = true, .imm_bits = 12, .imm_scale = 1, .features = &.{} };
    pub const ldr: A = .{ .mask = 0xFFC00000, .value = 0xF9400000, .rd_sp = false, .rn_sp = true, .imm_bits = 12, .imm_scale = 8, .features = &.{} };
    pub const str: A = .{ .mask = 0xFFC00000, .value = 0xF9000000, .rd_sp = false, .rn_sp = true, .imm_bits = 12, .imm_scale = 8, .features = &.{} };
    pub const ret: A = .{ .mask = 0xFFFFFC1F, .value = 0xD65F0000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const blr: A = .{ .mask = 0xFFFFFC1F, .value = 0xD63F0000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const b: A = .{ .mask = 0xFC000000, .value = 0x14000000, .rd_sp = false, .rn_sp = false, .imm_bits = 26, .imm_scale = 4, .features = &.{} };
    pub const bl: A = .{ .mask = 0xFC000000, .value = 0x94000000, .rd_sp = false, .rn_sp = false, .imm_bits = 26, .imm_scale = 4, .features = &.{} };
    pub const bcond: A = .{ .mask = 0xFF000010, .value = 0x54000000, .rd_sp = false, .rn_sp = false, .imm_bits = 19, .imm_scale = 4, .features = &.{} };
    pub const cbz: A = .{ .mask = 0xFF000000, .value = 0xB4000000, .rd_sp = false, .rn_sp = false, .imm_bits = 19, .imm_scale = 4, .features = &.{} };
    pub const cbnz: A = .{ .mask = 0xFF000000, .value = 0xB5000000, .rd_sp = false, .rn_sp = false, .imm_bits = 19, .imm_scale = 4, .features = &.{} };
    pub const sdiv: A = .{ .mask = 0xFFE0FC00, .value = 0x9AC00C00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const udiv: A = .{ .mask = 0xFFE0FC00, .value = 0x9AC00800, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const madd: A = .{ .mask = 0xFFE08000, .value = 0x9B000000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const msub: A = .{ .mask = 0xFFE08000, .value = 0x9B008000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const mneg: A = .{ .mask = 0xFFE0FC00, .value = 0x9B00FC00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const smulh: A = .{ .mask = 0xFFE0FC00, .value = 0x9B407C00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const umulh: A = .{ .mask = 0xFFE0FC00, .value = 0x9BC07C00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const neg: A = .{ .mask = 0xFF2003E0, .value = 0xCB0003E0, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const and_: A = .{ .mask = 0xFF200000, .value = 0x8A000000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const orr: A = .{ .mask = 0xFF200000, .value = 0xAA000000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const eor: A = .{ .mask = 0xFF200000, .value = 0xCA000000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const csel: A = .{ .mask = 0xFFE00C00, .value = 0x9A800000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const cset: A = .{ .mask = 0xFFFF0FE0, .value = 0x9A9F07E0, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const lsl_imm: A = .{ .mask = 0xFFC00000, .value = 0xD3400000, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const lsr_imm: A = .{ .mask = 0xFFC0FC00, .value = 0xD340FC00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
    pub const asr_imm: A = .{ .mask = 0xFFC0FC00, .value = 0x9340FC00, .rd_sp = false, .rn_sp = false, .imm_bits = 0, .imm_scale = 1, .features = &.{} };
};

const x_features_prev = [_][]const u8{ "3DNOW", "3DNOW2", "ADX", "AESKLE", "AESNI", "AMX_AVX512", "AMX_BF16", "AMX_COMPLEX", "AMX_FP16", "AMX_FP8", "AMX_INT8", "AMX_MOVRS", "AMX_TF32", "AMX_TILE", "AVX", "AVX10_2", "AVX2", "AVX512_BF16", "AVX512_BITALG", "AVX512_BW", "AVX512_CD", "AVX512_DQ", "AVX512_F", "AVX512_FP16", "AVX512_IFMA", "AVX512_VBMI", "AVX512_VBMI2", "AVX512_VNNI", "AVX512_VP2INTERSECT", "AVX512_VPOPCNTDQ", "AVX_IFMA", "AVX_NE_CONVERT", "AVX_VNNI", "AVX_VNNI_INT16", "AVX_VNNI_INT8", "BMI", "BMI2", "CET_IBT", "CET_SS", "CLDEMOTE", "CLFLUSH", "CLFLUSHOPT", "CLWB", "CLZERO", "CMOV", "CMPCCXADD", "CMPXCHG16B", "CMPXCHG8B", "ENQCMD", "F16C", "FMA", "FMA4", "FPU", "FSGSBASE", "FXSR", "GEODE", "GFNI", "HRESET", "I486", "INVLPGB", "KL", "LAHFSAHF", "LWP", "LZCNT", "MCOMMIT", "MMX", "MMX2", "MONITOR", "MONITORX", "MOVBE", "MOVDIR64B", "MOVDIRI", "MOVRS", "MSR", "MSRLIST", "MSR_IMM", "OSPKE", "PCLMULQDQ", "PCONFIG", "POPCNT", "PREFETCHI", "PREFETCHW", "PREFETCHWT1", "PTWRITE", "RAO_INT", "RDPID", "RDPRU", "RDRAND", "RDSEED", "RDTSC", "RDTSCP", "RMPQUERY", "RTM", "SEAM", "SERIALIZE", "SEV_ES", "SEV_SNP", "SHA", "SHA512", "SKINIT", "SM3", "SM4", "SMAP", "SMX", "SSE", "SSE2", "SSE3", "SSE4A", "SSE4_1", "SSE4_2", "SSSE3", "SVM", "TBM", "TSE", "TSXLDTRK", "UINTR", "USER_MSR", "VAES", "VMX", "VPCLMULQDQ", "WAITPKG", "WBNOINVD", "WRMSRNS", "XOP", "XSAVE", "XSAVEC", "XSAVEOPT", "XSAVES" };
const a_features_prev = [_][]const u8{ "AES", "ASIMD", "BF16", "BRBE", "BTI", "CHK", "CLRBHB", "CPA", "CRC32", "CSSC", "D128", "DGH", "DOTPROD", "FAMINMAX", "FCMA", "FHM", "FLAGM", "FLAGM2", "FP16", "FP8", "FP8DOT2", "FP8DOT4", "FP8FMA", "FRINTTS", "GCS", "HBC", "I8MM", "ITE", "JSCVT", "LOR", "LRCPC", "LRCPC2", "LRCPC3", "LS64", "LS64_ACCDATA", "LS64_V", "LSE", "LUT", "MOPS", "MTE", "MTE2", "PAUTH", "RAS", "RDM", "RPRFM", "SB", "SHA1", "SHA256", "SHA3", "SHA512", "SM3", "SM4", "SME", "SPE", "SPECRES", "SPECRES2", "SVE", "SVE2", "SVE2_AES", "SVE2_BITPERM", "SVE2_SHA3", "SVE2_SM4", "SVE_BF16", "SVE_F32MM", "SVE_F64MM", "SVE_I8MM", "SYSINSTR128", "SYSREG128", "THE", "TME", "TRF", "WFXT" };

fn expectX(want: X, got: x_isa.Form) !void {
    try t.expectEqualSlices(u8, want.opcode, got.opcode);
    try t.expectEqual(want.plus, got.plus);
    try t.expectEqual(want.modrm, got.modrm);
    try t.expectEqual(want.digit, got.digit);
    try t.expectEqual(want.imm, got.imm);
    try t.expectEqualSlices(x_isa.Feature, want.features, got.features);
}

fn expectXList(want: []const X, got: []const x_isa.Form) !void {
    try t.expectEqual(want.len, got.len);
    for (want, got) |w, g| try expectX(w, g);
}

test "x86_64 isa (LLVM): every entry encodes as in the previous table" {
    inline for (@typeInfo(x_prev).@"struct".decl_names) |name| {
        const want = @field(x_prev, name);
        const got = @field(x_isa.table, name);
        if (@TypeOf(want[0]) == X) {
            try expectXList(&want, &got);
        } else {
            try t.expectEqual(want.len, got.len);
            for (want, got) |w, g| try expectXList(w, g);
        }
    }
    // and no entry was added or dropped
    try t.expectEqual(@typeInfo(x_prev).@"struct".decl_names.len, @typeInfo(x_isa.table).@"struct".decl_names.len);
}

test "aarch64 isa (LLVM): every entry encodes as in the previous table, except add/sub/cmp imm" {
    inline for (@typeInfo(a_prev).@"struct".decl_names) |name| {
        const w = @field(a_prev, name);
        const g = @field(a_isa.table, name);
        // Reviewed change: LLVM's addsub_shifted_imm64 operand owns bits 23:22 (shift), so bit 23 is
        // no longer a fixed bit; the encoders still emit 0 there (checked below).
        const changed = comptime std.mem.eql(u8, name, "add_imm") or std.mem.eql(u8, name, "sub_imm") or std.mem.eql(u8, name, "cmp_imm");
        try t.expectEqual(if (changed) w.mask & ~@as(u32, 1 << 23) else w.mask, g.mask);
        try t.expectEqual(w.value, g.value);
        try t.expectEqual(w.rd_sp, g.rd_sp);
        try t.expectEqual(w.rn_sp, g.rn_sp);
        try t.expectEqual(w.imm_bits, g.imm_bits);
        try t.expectEqual(w.imm_scale, g.imm_scale);
        try t.expectEqualSlices(a_isa.Feature, w.features, g.features);
    }
    try t.expectEqual(@typeInfo(a_prev).@"struct".decl_names.len, @typeInfo(a_isa.table).@"struct".decl_names.len);
}

test "isa (LLVM): Feature names are unchanged (public API of target_features/deny_features)" {
    const xf = @typeInfo(x_isa.Feature).@"enum".field_names;
    try t.expectEqual(x_features_prev.len, xf.len);
    for (xf, 0..) |f, i| try t.expectEqualStrings(x_features_prev[i], f);
    const af = @typeInfo(a_isa.Feature).@"enum".field_names;
    try t.expectEqual(a_features_prev.len, af.len);
    for (af, 0..) |f, i| try t.expectEqualStrings(a_features_prev[i], f);
}

test "isa (LLVM): records and syntax come from LLVM" {
    try t.expectEqualStrings("ADD64ri8", x_isa.table.alu_ri[0][0].llvm);
    try t.expectEqualStrings("add $src1, $src2", x_isa.table.alu_ri[0][0].syntax);
    try t.expectEqualStrings("JCC_4", x_isa.table.jcc[15][1].llvm);
    try t.expectEqualStrings("jg $dst", x_isa.table.jcc[15][1].syntax);
    try t.expectEqualStrings("ORRXrs", a_isa.table.mov.llvm); // alias: mov = orr Xd, xzr, Xm
    try t.expectEqualStrings("mov $dst, $src", a_isa.table.mov.syntax);
    try t.expectEqualStrings("CSINCXr", a_isa.table.cset.llvm);
}

test "aarch64 isa (LLVM): add/sub/cmp imm encoders keep bit 23 clear and match the new masks" {
    if (!opts.aarch64) return error.SkipZigTest;
    const m = @import("aarch64/assembler.zig");
    const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const as = m.AssemblerT(.{ .validate = true }).init(&cb);
    try as.addImm(.x0, .x1, 1);
    try as.subImm(.x3, .x4, 4095);
    try as.cmpImm(.x2, 4095);
    const code = cb.code();
    try t.expectEqual(@as(usize, 12), code.len);
    const forms = [_]a_isa.Form{ a_isa.table.add_imm, a_isa.table.sub_imm, a_isa.table.cmp_imm };
    const want = [_]u32{ 0x91000420, 0xD13FFC83, 0xF13FFC5F };
    for (forms, want, 0..) |f, w, i| {
        const word = std.mem.readInt(u32, code[i * 4 ..][0..4], .little);
        try t.expectEqual(w, word);
        try t.expectEqual(f.value, word & f.mask);
        try t.expectEqual(@as(u32, 0), word & (1 << 23));
    }
}
