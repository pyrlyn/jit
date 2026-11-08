//! Bounded stack scratch space with allocator fallback. Only scoped temporary data
//! may use it; returned machine code always belongs to the caller's CodeBuffer.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Scratch = struct {
    fixed: std.heap.FixedBufferAllocator,
    fallback: Allocator,

    pub fn init(buffer: []u8, fallback: Allocator) Scratch {
        return .{ .fixed = .init(buffer), .fallback = fallback };
    }
    pub fn allocator(s: *Scratch) Allocator {
        return .{ .ptr = s, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, n: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const s: *Scratch = @ptrCast(@alignCast(ctx));
        return s.fixed.allocator().rawAlloc(n, alignment, ra) orelse s.fallback.rawAlloc(n, alignment, ra);
    }
    fn owner(s: *Scratch, buf: []u8) Allocator {
        return if (s.fixed.ownsSlice(buf)) s.fixed.allocator() else s.fallback;
    }
    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) bool {
        const s: *Scratch = @ptrCast(@alignCast(ctx));
        return s.owner(buf).rawResize(buf, alignment, n, ra);
    }
    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const s: *Scratch = @ptrCast(@alignCast(ctx));
        return s.owner(buf).rawRemap(buf, alignment, n, ra);
    }
    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const s: *Scratch = @ptrCast(@alignCast(ctx));
        s.owner(buf).rawFree(buf, alignment, ra);
    }
};

test "scratch: aligned allocations grow from stack to fallback and clean up on failure" {
    const t = std.testing;
    const Case = struct {
        fn run(backing: Allocator) !void {
            var buf: [128]u8 align(16) = undefined;
            var scratch = Scratch.init(&buf, backing);
            const a = scratch.allocator();
            var list = try std.ArrayList(u64).initCapacity(a, 4);
            defer list.deinit(a);
            try list.append(a, 123);
            try t.expect(scratch.fixed.ownsSlice(std.mem.sliceAsBytes(list.allocatedSlice())));
            try list.ensureTotalCapacity(a, 256);
            try t.expect(!scratch.fixed.ownsSlice(std.mem.sliceAsBytes(list.allocatedSlice())));
            try t.expectEqual(@as(u64, 123), list.items[0]);
            const aligned = try a.alignedAlloc(u8, .@"16", 32);
            defer a.free(aligned);
            try t.expectEqual(@as(usize, 0), @intFromPtr(aligned.ptr) % 16);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, Case.run, .{});
}
