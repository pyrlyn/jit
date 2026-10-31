//! Optional safety layer shared by all backends: operand validation and emission hooks.
//!
//! Both are selected at comptime through `Options`, so a disabled layer compiles to nothing:
//! validation is guarded by `if (comptime opts.validate)` and hooks by `Hooks != void`.
//! Hooks are a plain struct type with optional methods (duck typing via `@hasDecl`), called
//! directly (inlined), never through function pointers.
//!
//! Hook interface (all methods optional):
//!   pub fn pre(self: *H, ev: PreEvent) Error!void   // before encoding; may veto with an error
//!   pub fn post(self: *H, ev: Event) void           // after encoding; sees the emitted bytes
const std = @import("std");

pub const Options = struct {
    /// Validate operands before encoding and return an error instead of emitting bad code.
    validate: bool = false,
    /// Hook type (struct with optional `pre`/`post`), or `void` for no hooks.
    Hooks: type = void,
    /// CPU-feature check for emitted instructions (needs `validate`):
    /// - `.none`: no check;
    /// - `.target`: required features must be in `target_features` (checked at compile time);
    /// - `.host`: required features must be present on the running CPU (CPUID / sysctl).
    cpu_check: CpuCheck = .none,
    /// Feature names from AsmJit's database (`isa.Feature`), e.g. `&.{ "SSE4_2", "AVX2" }`.
    target_features: []const []const u8 = &.{},
    /// Deny list (any mode, checked at compile time): an instruction that needs one of these
    /// features is a compile error, e.g. `&.{ "AVX512_F", "RTM" }`.
    deny_features: []const []const u8 = &.{},
};

pub const CpuCheck = enum { none, target, host };

/// Converts feature names to a backend's `Features` set at compile time.
pub fn featureSet(comptime Feature: type, comptime names: []const []const u8) std.EnumSet(Feature) {
    @setEvalBranchQuota(100_000);
    var set: std.EnumSet(Feature) = .empty;
    for (names) |n| set.insert(std.meta.stringToEnum(Feature, n) orelse @compileError("unknown CPU feature: " ++ n));
    return set;
}

/// Event passed to `pre` hooks.
pub fn PreEvent(comptime Inst: type, comptime Ops: type) type {
    return struct {
        id: Inst,
        ops: Ops,
        /// Byte offset where the instruction will start.
        offset: u32,
    };
}

/// Event passed to `post` hooks.
pub fn Event(comptime Inst: type, comptime Ops: type) type {
    return struct {
        id: Inst,
        ops: Ops,
        /// Byte offset of the first byte of the instruction.
        offset: u32,
        /// Encoded bytes (valid until the next emit into the buffer).
        bytes: []const u8,
    };
}

/// Counts instructions and bytes. Works with any backend.
pub const CountingHooks = struct {
    instructions: usize = 0,
    bytes: usize = 0,
    pub inline fn post(self: *CountingHooks, ev: anytype) void {
        self.instructions += 1;
        self.bytes += ev.bytes.len;
    }
};

/// Writes one line per instruction ("0004  mov_rr  48 89 c8") into a fixed buffer.
/// Output past the buffer end is dropped and `truncated` is set.
pub fn LogHooks(comptime cap: usize) type {
    return struct {
        const Self = @This();
        buf: [cap]u8 = undefined,
        len: usize = 0,
        truncated: bool = false,
        pub fn post(self: *Self, ev: anytype) void {
            self.print("{x:0>4}  {s:<8}", .{ ev.offset, @tagName(ev.id) });
            for (ev.bytes) |b| self.print(" {x:0>2}", .{b});
            self.print("\n", .{});
        }
        fn print(self: *Self, comptime fmt: []const u8, args: anytype) void {
            const s = std.fmt.bufPrint(self.buf[self.len..], fmt, args) catch {
                self.truncated = true;
                return;
            };
            self.len += s.len;
        }
        pub fn text(self: *const Self) []const u8 {
            return self.buf[0..self.len];
        }
    };
}

/// Adapter for runtime callbacks (e.g. from C or a plugin). Costs one indirect call per
/// instruction, only when this type is chosen as `Hooks`.
pub fn CallbackHooks(comptime Ev: type) type {
    return struct {
        ctx: ?*anyopaque = null,
        post_fn: ?*const fn (ctx: ?*anyopaque, ev: Ev) void = null,
        pub fn post(self: *@This(), ev: Ev) void {
            if (self.post_fn) |f| f(self.ctx, ev);
        }
    };
}
