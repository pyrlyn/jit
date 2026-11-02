//! x86-64 CPU feature checks: per-form features from the LLVM-generated vector table,
//! `cpu_check = .target/.host` on `vinst`, `-Dx86-features` parsing/gating and `CpuInfo`.
const std = @import("std");
const t = std.testing;
const x = @import("x86_64/assembler.zig");
const CodeBuffer = @import("core/code_buffer.zig").CodeBuffer;
const vec = x.vec;
const isa = x.isa;
const ptr = x.ptr;

test {
    _ = x.cpu;
}

fn formsNeed(comptime mn: vec.Mnemonic, f: isa.Feature) usize {
    var n: usize = 0;
    for (@field(vec.tbl.table, @tagName(mn))) |fm| for (fm.features) |g| {
        if (g == f) n += 1;
    };
    return n;
}

test "cpu features: vector forms carry LLVM predicates" {
    try t.expect(formsNeed(.addps, .SSE) > 0);
    try t.expect(formsNeed(.vaddps, .AVX) > 0);
    try t.expect(formsNeed(.vaddps, .AVX512_F) > 0);
    try t.expect(formsNeed(.vpaddb, .AVX512_BW) > 0);
    try t.expect(formsNeed(.vfmadd231ps, .FMA) > 0);
    try t.expect(formsNeed(.paddq, .SSE2) > 0);
    try t.expect(formsNeed(.pshufb, .SSSE3) > 0);
}

test "cpu features: cpu_check = .target selects per form" {
    const A = x.AssemblerT(.{ .validate = true, .cpu_check = .target, .target_features = &.{ "SSE", "SSE2", "AVX" } });
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = A.init(&cb);
    try a.vinst(.addps, .{ vec.xmm(0), vec.xmm(1) }, .{});
    try a.vinst(.vaddps, .{ vec.ymm(0), vec.ymm(1), vec.ymm(2) }, .{}); // VEX: AVX
    // zmm / xmm16 need EVEX (AVX512_F), outside the target set.
    try t.expectError(error.UnsupportedCpuFeature, a.vinst(.vaddps, .{ vec.zmm(0), vec.zmm(1), vec.zmm(2) }, .{}));
    try t.expectError(error.UnsupportedCpuFeature, a.vinst(.vaddps, .{ vec.xmm(16), vec.xmm(1), vec.xmm(2) }, .{}));
    try t.expectError(error.UnsupportedCpuFeature, a.vinst(.vpaddd, .{ vec.ymm(0), vec.ymm(1), vec.ymm(2) }, .{})); // AVX2
    // Rejected instructions emit nothing.
    try t.expectEqual(@as(usize, 3 + 4), cb.code().len);
    const B = x.AssemblerT(.{ .validate = true, .cpu_check = .target, .target_features = &.{ "AVX", "AVX2", "AVX512_F" } });
    try B.init(&cb).vinst(.vaddps, .{ vec.zmm(0), vec.zmm(1), vec.mem(ptr(.rax, 64), 64) }, .{});
    try B.init(&cb).vinst(.vpaddd, .{ vec.ymm(0), vec.ymm(1), vec.ymm(2) }, .{});
}

test "cpu features: cpu_check = .host follows CPUID" {
    const host = x.cpu.host() orelse return error.SkipZigTest;
    const A = x.AssemblerT(.{ .validate = true, .cpu_check = .host });
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    const a = A.init(&cb);
    try a.vinst(.addps, .{ vec.xmm(0), vec.xmm(1) }, .{}); // SSE: baseline
    const r = a.vinst(.vaddps, .{ vec.ymm(0), vec.ymm(1), vec.ymm(2) }, .{});
    if (host.contains(.AVX)) try r else try t.expectError(error.UnsupportedCpuFeature, r);
    const z = a.vinst(.vaddps, .{ vec.zmm(0), vec.zmm(1), vec.zmm(2) }, .{});
    if (host.contains(.AVX512_F)) try z else try t.expectError(error.UnsupportedCpuFeature, z);
}

test "cpu features: -Dx86-features parsing and form gating" {
    const g = comptime x.parseFeatureList("avx2, sse4_2,avx512f");
    try t.expect(g.contains(.AVX2) and g.contains(.SSE4_2) and g.contains(.AVX512_F));
    try t.expect(g.contains(.SSE2)); // baseline always included
    try t.expect(!g.contains(.AVX512_BW));
    const avx_only = comptime vec.gatedForms(.vaddps, x.parseFeatureList("avx"));
    try t.expect(avx_only.len > 0);
    for (avx_only) |f| try t.expect(f.enc != .evex);
    const all = @field(vec.tbl.table, "vaddps");
    try t.expect(avx_only.len < all.len);
    if (x.build_gate) |bg| try t.expect(bg.contains(.SSE2));
}
