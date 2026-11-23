//! Bounded stack scratch space with allocator fallback, and an exact-fit arena. Only scoped
//! temporary data may use them; returned machine code always belongs to the caller's CodeBuffer.
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

/// Arena for one compile: everything is released by `deinit`. A new block is as large as the
/// request (at least `min_block`), not 1.5x the previous one as in `std.heap.ArenaAllocator`:
/// the footprint follows the data, and it is live until the machine code is complete.
pub const Arena = struct {
    backing: Allocator,
    /// Block that serves small requests; older and dedicated blocks follow it.
    head: ?*Block = null,
    /// Bytes of `head` in use (including its header).
    end: usize = 0,

    const Block = struct { next: ?*Block, len: usize };
    const block_align: std.mem.Alignment = .of(Block);
    const min_block = 1024;

    pub fn init(backing: Allocator) Arena {
        return .{ .backing = backing };
    }
    pub fn deinit(a: *Arena) void {
        var it = a.head;
        while (it) |b| {
            it = b.next;
            a.backing.rawFree(@as([*]u8, @ptrCast(b))[0..b.len], block_align, @returnAddress());
        }
        a.* = undefined;
    }
    pub fn allocator(a: *Arena) Allocator {
        return .{ .ptr = a, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, n: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const a: *Arena = @ptrCast(@alignCast(ctx));
        if (a.head) |b| {
            const base = @intFromPtr(b);
            const at = alignment.forward(base + a.end) - base;
            if (at <= b.len and b.len - at >= n) {
                a.end = at + n;
                return @ptrFromInt(base + at);
            }
        }
        // Worst-case padding after the header keeps any alignment.
        const pad = alignment.toByteUnits() -| block_align.toByteUnits();
        const need = std.math.add(usize, @sizeOf(Block) + pad, n) catch return null;
        const len = @max(need, min_block);
        const raw = a.backing.rawAlloc(len, block_align, ra) orelse return null;
        const b: *Block = @ptrCast(@alignCast(raw));
        const base = @intFromPtr(b);
        const at = alignment.forward(base + @sizeOf(Block)) - base;
        // A large request gets a block of its own behind `head`, whose free space stays usable.
        if (a.head != null and len == need and a.head.?.len - a.end >= min_block / 8) {
            b.* = .{ .next = a.head.?.next, .len = len };
            a.head.?.next = b;
        } else {
            b.* = .{ .next = a.head, .len = len };
            a.head = b;
            a.end = at + n;
        }
        return @ptrFromInt(base + at);
    }
    /// In place only for the latest allocation of `head` (a growing list).
    fn resize(ctx: *anyopaque, buf: []u8, _: std.mem.Alignment, n: usize, _: usize) bool {
        const a: *Arena = @ptrCast(@alignCast(ctx));
        if (n <= buf.len) return true;
        const b = a.head orelse return false;
        const base = @intFromPtr(b);
        if (@intFromPtr(buf.ptr) + buf.len != base + a.end) return false;
        if (b.len - a.end < n - buf.len) return false;
        a.end += n - buf.len;
        return true;
    }
    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return if (resize(ctx, buf, alignment, n, ra)) buf.ptr else null;
    }
    fn free(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {}
};

test "arena: blocks fit the requests, alignment and in-place growth" {
    const t = std.testing;
    const Case = struct {
        fn run(backing: Allocator) !void {
            var arena = Arena.init(backing);
            defer arena.deinit();
            const a = arena.allocator();
            const small = try a.alloc(u8, 3);
            @memset(small, 1);
            const wide = try a.alignedAlloc(u8, .@"64", 100);
            try t.expectEqual(@as(usize, 0), @intFromPtr(wide.ptr) % 64);
            @memset(wide, 2);
            // A large request leaves the current block in use.
            const big = try a.alloc(u64, 4096);
            @memset(big, 3);
            const after = try a.alloc(u8, 8);
            try t.expect(@intFromPtr(after.ptr) > @intFromPtr(wide.ptr) and @intFromPtr(after.ptr) - @intFromPtr(wide.ptr) < Arena.min_block);
            // The latest allocation grows in place; an earlier one moves.
            var list: std.ArrayList(u32) = .empty;
            for (0..600) |i| try list.append(a, @intCast(i));
            for (list.items, 0..) |v, i| try t.expectEqual(@as(u32, @intCast(i)), v);
            try t.expect(a.resize(small, 2));
            try t.expect(!a.resize(small, 4096));
            try t.expectEqual(@as(u8, 1), small[0]);
            try t.expectEqual(@as(u8, 2), wide[99]);
            try t.expectEqual(@as(u64, 3), big[4095]);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, Case.run, .{});
}

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
