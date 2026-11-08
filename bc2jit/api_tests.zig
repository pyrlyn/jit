const std = @import("std");
const api = @import("bc2jit");
const t = std.testing;
const add_v0 = "KXBC\x00\x02\x06\x00\x02\x00\x02\x01\x10\xf0";
const add_v1 = "KXBC\x01\x02\x06\x00\x02\x00\x02\x01\x10\xf0";

test "public API: optional support, validation, ownership and execution" {
    var diag: api.Diagnostic = .{ .offset = 99 };
    if (!api.enabled) {
        try t.expectError(error.FeatureDisabled, api.compile(t.allocator, .aarch64, add_v0, .{}, &diag));
        try t.expectEqual(@as(usize, 0), diag.offset);
        return;
    }
    try t.expectError(error.BadMagic, api.compile(t.allocator, .aarch64, "BAD!\x00\x00\x00\x00", .{}, &diag));
    try t.expectEqualStrings("not a KXBC module (bad magic)", api.errorMessage(error.BadMagic));
    try t.expectEqualStrings("out of memory", api.errorMessage(error.OutOfMemory));
    for ([_][]const u8{ add_v0, add_v1 }) |bytes| {
        for (std.meta.tags(api.Target)) |target| {
            if (!target.isBuilt()) {
                try t.expectError(error.BackendNotBuilt, api.compile(t.allocator, target, bytes, .{}, &diag));
                continue;
            }
            // The input allocation can be freed before using code or entry.
            const input = try t.allocator.dupe(u8, bytes);
            var c = api.compile(t.allocator, target, input, .{ .executable = false }, &diag) catch |e| {
                t.allocator.free(input);
                return e;
            };
            t.allocator.free(input);
            defer c.deinit();
            try t.expect(c.code().len > 0);
            try t.expect(c.exec == null and c.entry() == null);
            var executable = try api.compile(t.allocator, target, bytes, .{}, &diag);
            defer executable.deinit();
            try t.expectEqualSlices(u8, c.code(), executable.code());
            const runnable = if (bytes[4] == 0) target.canExecute() else api.Target.host() == target;
            try t.expectEqual(runnable, executable.entry() != null);
            if (executable.entry()) |f| try t.expectEqual(@as(i64, 42), f(40, 2, 0, 0));
        }
    }
}

fn allocationCase(gpa: std.mem.Allocator) !void {
    var diag: api.Diagnostic = .{};
    for ([_][]const u8{ add_v0, add_v1 }) |bytes| {
        var c = try api.compile(gpa, .riscv64, bytes, .{ .executable = false }, &diag);
        defer c.deinit();
    }
}

test "public API: releases allocations on every failure" {
    if (!api.enabled) return error.SkipZigTest;
    try t.checkAllAllocationFailures(t.allocator, allocationCase, .{});
}
