//! Safety layer demo: validation errors and emission hooks (logging + counting + custom).
//! Generates x86-64 and AArch64 code on any host (nothing is executed).
//!   zig build run-example_hooks
const std = @import("std");
const pjit = @import("pyrlyn_jit");
const x = pjit.x86_64;
const a64 = pjit.aarch64;

/// A custom hook: counts instructions per id and logs them through LogHooks.
const Stats = struct {
    per_id: std.EnumArray(x.Inst, u32) = .initFill(0),
    log: x.LogHooks(1024) = .{},
    pub fn post(self: *Stats, ev: x.Event) void {
        self.per_id.getPtr(ev.id).* += 1;
        self.log.post(ev);
    }
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    var cb = pjit.CodeBuffer.init(gpa.allocator());
    defer cb.deinit();

    // x86-64 with validation and a custom hook.
    var stats: Stats = .{};
    const a = x.AssemblerT(.{ .validate = true, .Hooks = Stats }).initHooks(&cb, &stats);
    try a.push(.rbp);
    try a.movRR(.rbp, .rsp);
    try a.movRM(.rax, x.ptrIdx(.rdi, .rcx, .x8, 16));
    try a.addRI(.rax, 1);
    try a.pop(.rbp);
    try a.ret();
    std.debug.print("x86-64 log:\n{s}", .{stats.log.text()});
    std.debug.print("push={d} ret={d}\n", .{ stats.per_id.get(.push), stats.per_id.get(.ret) });

    // Validation: rsp as index is unencodable -> error instead of silently wrong bytes.
    if (a.movRM(.rax, x.ptrIdx(.rdi, .rsp, .x1, 0))) |_| unreachable else |err| std.debug.print("rejected [rdi+rsp]: {s}\n", .{@errorName(err)});

    // AArch64 with a counting hook.
    cb.reset();
    var count: a64.CountingHooks = .{};
    const b = a64.AssemblerT(.{ .validate = true, .Hooks = a64.CountingHooks }).initHooks(&cb, &count);
    try b.movImm(.x0, 0xDEAD_BEEF_0000_0001);
    try b.subImm(.zr_sp, .zr_sp, 16);
    try b.ret();
    std.debug.print("aarch64: {d} instructions, {d} bytes\n", .{ count.instructions, count.bytes });
    if (b.add(.x0, .zr_sp, .x1)) |_| unreachable else |err| std.debug.print("rejected add x0, sp(xzr), x1: {s}\n", .{@errorName(err)});
}
