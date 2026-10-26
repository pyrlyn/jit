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

pub const Error = error{ OutOfMemory, UnboundLabel, LabelAlreadyBound, DisplacementOutOfRange, InvalidLabel, CodeTooLarge, TooManyLabels, EmitFailed };

/// Code is limited to 2 GiB so every rel32 / label offset fits in an i32 / u32.
pub const max_code_size: usize = std.math.maxInt(i32);

const unbound: u32 = std.math.maxInt(u32);

pub const ErrorHandler = struct {
    ctx: ?*anyopaque = null,
    func: *const fn (ctx: ?*anyopaque, err: anyerror, offset: u32) void,
};

pub const CodeBuffer = struct {
    gpa: Allocator,
    data: []u8 = &.{},
    len: usize = 0,
    labels: std.ArrayList(u32) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,
    /// Limits (set after `init`): code size in bytes (<= `max_code_size`) and label count.
    max_size: usize = max_code_size,
    max_labels: u32 = std.math.maxInt(u32),
    /// First error recorded by `keep` (sticky error); `finalize` then fails with `EmitFailed`.
    sticky: ?anyerror = null,
    /// Optional error handler, called by `keep` for every recorded error with the code offset.
    error_handler: ?ErrorHandler = null,

    inline fn limit(self: *const CodeBuffer) usize {
        return @min(self.max_size, max_code_size);
    }

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
        self.sticky = null;
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

    noinline fn grow(self: *CodeBuffer, n: usize) Error!void {
        @branchHint(.cold);
        if (self.len + n > self.limit()) return error.CodeTooLarge;
        var cap = @max(self.data.len * 2, 4096);
        while (cap - self.len < n) cap *= 2;
        cap = @min(cap, self.limit() + 16); // +16: room for the 16-byte instruction store
        if (self.gpa.resize(self.data, cap)) {
            self.data = self.data.ptr[0..cap];
            return;
        }
        const nd = try self.gpa.alloc(u8, cap);
        @memcpy(nd[0..self.len], self.data[0..self.len]);
        self.gpa.free(self.data);
        self.data = nd;
    }

    /// Sticky-error style emission without `try` at every call:
    ///     cb.keep(a.addRR(.rax, .rcx));
    ///     cb.keep(a.inst(.mov, .{ ... }));
    ///     try cb.finalize(); // EmitFailed if anything failed; cb.sticky has the first error
    /// Every error is passed to `error_handler` (if set); only the first is kept.
    pub fn keep(self: *CodeBuffer, result: anytype) void {
        if (result) |_| {} else |e| {
            @branchHint(.cold);
            if (self.error_handler) |h| h.func(h.ctx, e, self.offset());
            if (self.sticky == null) self.sticky = e;
        }
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
    /// Fast path inlined into the caller; growth is a cold out-of-line call.
    pub inline fn newLabel(self: *CodeBuffer) Error!Label {
        const id: u32 = @intCast(self.labels.items.len);
        if (id >= self.max_labels) return error.TooManyLabels;
        if (self.labels.items.len == self.labels.capacity) try self.growLabels();
        self.labels.appendAssumeCapacity(unbound);
        return .{ .id = id };
    }
    noinline fn growLabels(self: *CodeBuffer) Error!void {
        @branchHint(.cold);
        try self.labels.ensureUnusedCapacity(self.gpa, @max(self.labels.capacity, 64));
    }

    /// Errors: `InvalidLabel` (not created by this buffer), `LabelAlreadyBound` (bound twice).
    pub inline fn bind(self: *CodeBuffer, l: Label) Error!void {
        if (l.id >= self.labels.items.len) return error.InvalidLabel;
        const p = &self.labels.items[l.id];
        if (p.* != unbound) return error.LabelAlreadyBound;
        p.* = self.offset();
    }

    pub inline fn labelOffset(self: *const CodeBuffer, l: Label) ?u32 {
        const o = self.labels.items[l.id];
        return if (o == unbound) null else o;
    }

    pub inline fn addFixup(self: *CodeBuffer, f: Fixup) Error!void {
        if (self.fixups.items.len == self.fixups.capacity) try self.growFixups();
        self.fixups.appendAssumeCapacity(f);
    }
    noinline fn growFixups(self: *CodeBuffer) Error!void {
        @branchHint(.cold);
        try self.fixups.ensureUnusedCapacity(self.gpa, @max(self.fixups.capacity, 64));
    }

    /// True if `l` was created by this buffer.
    pub inline fn isValidLabel(self: *const CodeBuffer, l: Label) bool {
        return l.id < self.labels.items.len;
    }

    /// Labels that are referenced by a pending fixup but not bound (deduplicated, in order of
    /// first use). Writes up to `out.len` ids; use after `finalize` returned `UnboundLabel`.
    pub fn unresolvedLabels(self: *const CodeBuffer, out: []u32) []u32 {
        var n: usize = 0;
        outer: for (self.fixups.items) |f| {
            if (self.labels.items[f.label] != unbound) continue;
            for (out[0..n]) |id| if (id == f.label) continue :outer;
            if (n == out.len) break;
            out[n] = f.label;
            n += 1;
        }
        return out[0..n];
    }

    /// Resolves all label fixups. Call once after emission, before copying code out.
    /// Checks are always on: an unbound label returns `UnboundLabel` (see `unresolvedLabels`),
    /// a displacement that does not fit its field returns `DisplacementOutOfRange`, and code
    /// over 2 GiB returns `CodeTooLarge`. Nothing is patched when an error is returned.
    pub fn finalize(self: *CodeBuffer) Error!void {
        if (self.sticky != null) return error.EmitFailed;
        if (self.len > self.limit()) return error.CodeTooLarge;
        // Pass 1: check everything, so a failing finalize leaves the code untouched.
        for (self.fixups.items) |f| {
            const target = self.labels.items[f.label];
            if (target == unbound) return error.UnboundLabel;
            const disp: i64 = @as(i64, target) - @as(i64, f.base);
            const ok = switch (f.kind) {
                .rel8 => std.math.cast(i8, disp) != null,
                .rel32 => std.math.cast(i32, disp) != null,
                .a64_imm26 => @mod(disp, 4) == 0 and @divExact(disp, 4) >= -(1 << 25) and @divExact(disp, 4) < (1 << 25),
                .a64_imm19 => @mod(disp, 4) == 0 and @divExact(disp, 4) >= -(1 << 18) and @divExact(disp, 4) < (1 << 18),
            };
            if (!ok) return error.DisplacementOutOfRange;
        }
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
