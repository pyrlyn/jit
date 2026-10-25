//! Arch-agnostic code buffer: bytes + labels + label fixups.
//! Equivalent of AsmJit's CodeHolder (single .text section) + CodeBuffer.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Label = struct { id: u32 };

pub const FixupKind = enum(u8) {
    /// x86: signed 8-bit displacement relative to end of instruction.
    rel8,
    /// x86: signed 32-bit displacement relative to end of instruction.
    rel32,
    /// AArch64 B/BL: imm26 * 4, relative to instruction start.
    a64_imm26,
    /// AArch64 B.cond/CBZ/CBNZ: imm19 * 4 at bits [23:5], relative to instruction start.
    a64_imm19,
};

pub const Fixup = struct {
    /// Offset of the field (x86) or of the instruction word (a64).
    at: u32,
    /// Offset the displacement is relative to.
    base: u32,
    label: u32,
    kind: FixupKind,
};

pub const Error = error{ OutOfMemory, UnboundLabel, LabelAlreadyBound, DisplacementOutOfRange };

const unbound: u32 = std.math.maxInt(u32);

pub const CodeBuffer = struct {
    gpa: Allocator,
    data: []u8 = &.{},
    len: usize = 0,
    labels: std.ArrayList(u32) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,

    pub fn init(gpa: Allocator) CodeBuffer {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *CodeBuffer) void {
        self.gpa.free(self.data);
        self.labels.deinit(self.gpa);
        self.fixups.deinit(self.gpa);
        self.* = undefined;
    }

    /// Drops code, labels and fixups but keeps allocated capacity (cheap reuse).
    pub fn reset(self: *CodeBuffer) void {
        self.len = 0;
        self.labels.clearRetainingCapacity();
        self.fixups.clearRetainingCapacity();
    }

    pub fn code(self: *const CodeBuffer) []const u8 {
        return self.data[0..self.len];
    }

    pub inline fn offset(self: *const CodeBuffer) u32 {
        return @intCast(self.len);
    }

    /// Makes sure `n` more bytes can be written without reallocation.
    pub inline fn ensure(self: *CodeBuffer, n: usize) Error!void {
        if (self.data.len - self.len < n) return self.grow(n);
    }

    fn grow(self: *CodeBuffer, n: usize) Error!void {
        var cap = @max(self.data.len * 2, 4096);
        while (cap - self.len < n) cap *= 2;
        if (self.gpa.resize(self.data, cap)) {
            self.data = self.data.ptr[0..cap];
            return;
        }
        const nd = try self.gpa.alloc(u8, cap);
        @memcpy(nd[0..self.len], self.data[0..self.len]);
        self.gpa.free(self.data);
        self.data = nd;
    }

    // Unchecked writers: caller must have called ensure().
    pub inline fn put8(self: *CodeBuffer, b: u8) void {
        self.data[self.len] = b;
        self.len += 1;
    }
    pub inline fn put16(self: *CodeBuffer, v: u16) void {
        std.mem.writeInt(u16, self.data[self.len..][0..2], v, .little);
        self.len += 2;
    }
    pub inline fn put32(self: *CodeBuffer, v: u32) void {
        std.mem.writeInt(u32, self.data[self.len..][0..4], v, .little);
        self.len += 4;
    }
    pub inline fn put64(self: *CodeBuffer, v: u64) void {
        std.mem.writeInt(u64, self.data[self.len..][0..8], v, .little);
        self.len += 8;
    }

    pub fn emitBytes(self: *CodeBuffer, bytes: []const u8) Error!void {
        try self.ensure(bytes.len);
        @memcpy(self.data[self.len..][0..bytes.len], bytes);
        self.len += bytes.len;
    }

    // ---- labels -------------------------------------------------------------
    pub fn newLabel(self: *CodeBuffer) Error!Label {
        const id: u32 = @intCast(self.labels.items.len);
        try self.labels.append(self.gpa, unbound);
        return .{ .id = id };
    }

    pub fn bind(self: *CodeBuffer, l: Label) Error!void {
        const p = &self.labels.items[l.id];
        if (p.* != unbound) return error.LabelAlreadyBound;
        p.* = self.offset();
    }

    pub inline fn labelOffset(self: *const CodeBuffer, l: Label) ?u32 {
        const o = self.labels.items[l.id];
        return if (o == unbound) null else o;
    }

    pub fn addFixup(self: *CodeBuffer, f: Fixup) Error!void {
        try self.fixups.append(self.gpa, f);
    }

    /// Resolves all label fixups. Call once after emission, before copying code out.
    pub fn finalize(self: *CodeBuffer) Error!void {
        for (self.fixups.items) |f| {
            const target = self.labels.items[f.label];
            if (target == unbound) return error.UnboundLabel;
            const disp: i64 = @as(i64, target) - @as(i64, f.base);
            const p = self.data[f.at..];
            switch (f.kind) {
                .rel8 => p[0] = @bitCast(std.math.cast(i8, disp) orelse return error.DisplacementOutOfRange),
                .rel32 => std.mem.writeInt(i32, p[0..4], std.math.cast(i32, disp) orelse return error.DisplacementOutOfRange, .little),
                .a64_imm26, .a64_imm19 => {
                    const words = @divExact(disp, 4);
                    var w = std.mem.readInt(u32, p[0..4], .little);
                    if (f.kind == .a64_imm26) {
                        if (words < -(1 << 25) or words >= (1 << 25)) return error.DisplacementOutOfRange;
                        w |= @as(u32, @truncate(@as(u64, @bitCast(words)))) & 0x03FF_FFFF;
                    } else {
                        if (words < -(1 << 18) or words >= (1 << 18)) return error.DisplacementOutOfRange;
                        w |= (@as(u32, @truncate(@as(u64, @bitCast(words)))) & 0x7FFFF) << 5;
                    }
                    std.mem.writeInt(u32, p[0..4], w, .little);
                },
            }
        }
        self.fixups.clearRetainingCapacity();
    }
};

test "labels and rel32 fixup" {
    var cb = CodeBuffer.init(std.testing.allocator);
    defer cb.deinit();
    const l = try cb.newLabel();
    try cb.ensure(5);
    cb.put8(0xE9);
    try cb.addFixup(.{ .at = 1, .base = 5, .label = l.id, .kind = .rel32 });
    cb.put32(0);
    try cb.bind(l);
    try cb.finalize();
    try std.testing.expectEqualSlices(u8, &.{ 0xE9, 0, 0, 0, 0 }, cb.code());
}
