// Generated call helpers h0..h16 (C ABI): mix of the arguments and a call counter.
pub var calls: i64 = 0;
fn mix(acc: i64, a: i64, i: i64) i64 {
    const r = @as(u64, @bitCast(acc));
    return @as(i64, @bitCast((r << 7) | (r >> 57))) ^ (a *% (2 * i + 1));
}
pub fn h0() callconv(.c) i64 {
    calls +%= 1;
    const acc: i64 = 0x1234 + 0 + calls *% 1000003;
    return acc;
}
pub fn h1(a0: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 1 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    return acc;
}
pub fn h2(a0: i64, a1: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 2 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    return acc;
}
pub fn h3(a0: i64, a1: i64, a2: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 3 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    return acc;
}
pub fn h4(a0: i64, a1: i64, a2: i64, a3: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 4 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    return acc;
}
pub fn h5(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 5 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    return acc;
}
pub fn h6(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 6 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    return acc;
}
pub fn h7(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 7 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    return acc;
}
pub fn h8(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 8 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    return acc;
}
pub fn h9(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 9 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    return acc;
}
pub fn h10(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 10 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    return acc;
}
pub fn h11(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 11 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    acc = mix(acc, a10, 10);
    return acc;
}
pub fn h12(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64, a11: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 12 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    acc = mix(acc, a10, 10);
    acc = mix(acc, a11, 11);
    return acc;
}
pub fn h13(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64, a11: i64, a12: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 13 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    acc = mix(acc, a10, 10);
    acc = mix(acc, a11, 11);
    acc = mix(acc, a12, 12);
    return acc;
}
pub fn h14(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64, a11: i64, a12: i64, a13: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 14 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    acc = mix(acc, a10, 10);
    acc = mix(acc, a11, 11);
    acc = mix(acc, a12, 12);
    acc = mix(acc, a13, 13);
    return acc;
}
pub fn h15(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64, a11: i64, a12: i64, a13: i64, a14: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 15 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    acc = mix(acc, a10, 10);
    acc = mix(acc, a11, 11);
    acc = mix(acc, a12, 12);
    acc = mix(acc, a13, 13);
    acc = mix(acc, a14, 14);
    return acc;
}
pub fn h16(a0: i64, a1: i64, a2: i64, a3: i64, a4: i64, a5: i64, a6: i64, a7: i64, a8: i64, a9: i64, a10: i64, a11: i64, a12: i64, a13: i64, a14: i64, a15: i64) callconv(.c) i64 {
    calls +%= 1;
    var acc: i64 = 0x1234 + 16 + calls *% 1000003;
    acc = mix(acc, a0, 0);
    acc = mix(acc, a1, 1);
    acc = mix(acc, a2, 2);
    acc = mix(acc, a3, 3);
    acc = mix(acc, a4, 4);
    acc = mix(acc, a5, 5);
    acc = mix(acc, a6, 6);
    acc = mix(acc, a7, 7);
    acc = mix(acc, a8, 8);
    acc = mix(acc, a9, 9);
    acc = mix(acc, a10, 10);
    acc = mix(acc, a11, 11);
    acc = mix(acc, a12, 12);
    acc = mix(acc, a13, 13);
    acc = mix(acc, a14, 14);
    acc = mix(acc, a15, 15);
    return acc;
}
pub fn addr(n: usize) usize {
    return switch (n) {
        0 => @intFromPtr(&h0),
        1 => @intFromPtr(&h1),
        2 => @intFromPtr(&h2),
        3 => @intFromPtr(&h3),
        4 => @intFromPtr(&h4),
        5 => @intFromPtr(&h5),
        6 => @intFromPtr(&h6),
        7 => @intFromPtr(&h7),
        8 => @intFromPtr(&h8),
        9 => @intFromPtr(&h9),
        10 => @intFromPtr(&h10),
        11 => @intFromPtr(&h11),
        12 => @intFromPtr(&h12),
        13 => @intFromPtr(&h13),
        14 => @intFromPtr(&h14),
        15 => @intFromPtr(&h15),
        16 => @intFromPtr(&h16),
        else => unreachable,
    };
}
