//! Tiered JIT engine (`pyrlyn_jit.engine`): stable entry points, profiling tier-0 code,
//! automatic tier-up to optimized code with runtime inlining.
//!
//! Every function gets an immutable trampoline (`address`) that loads the current code address
//! from a heap slot and jumps to it; switching tiers is a release store into the slot after the
//! new code is executable. Code is never patched and never freed before `destroy`, so threads
//! still running an old tier finish it safely.
//!
//! Tier 0 is the function instrumented by `opt/profile.zig` (block and edge counters; with
//! `auto_tier_up`, a call into `hookEntry` when the call count reaches `hot_calls`). Tier 1 is
//! `opt.optimize` of the original IR with the collected profile and a resolver that maps engine
//! trampoline addresses back to the callee's IR, so calls between engine functions are inlined.
//!
//! The engine keeps its own copy of each function with one extra entry block prepended
//! (`br`/`bra` to the original block 0), so stored block `b + 1` is the caller's block `b` and
//! counter 0 counts calls even when the original block 0 is a loop header. `profile` returns
//! counters in the caller's block numbering.
//!
//! Compiles (tier 0 and tier 1) are serialized by a spin lock; generated code may be called from
//! any number of threads at any time.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("pyrlyn_jit_options");
const ir = @import("ir.zig");
const opt = @import("opt/opt.zig");
const profiling = @import("opt/profile.zig");
const CodeBuffer = @import("code_buffer.zig").CodeBuffer;
const JitRuntime = @import("jit_runtime.zig");

/// IR Compiler for the host (void when the host has none or it is disabled at build time).
const host = switch (builtin.cpu.arch) {
    .aarch64 => if (build_options.aarch64) @import("../aarch64/compiler.zig") else void,
    .x86_64 => if (build_options.x86_64) @import("../x86_64/compiler.zig") else void,
    else => void,
};

/// Whether this build can JIT for the host; otherwise `Engine.create` fails with
/// `error.UnsupportedHost`.
pub const supported = host != void;

/// Trampoline encoding. Each trampoline jumps through its slot using only a scratch register
/// that is call-clobbered and never carries an argument (x16 = IP0 on AArch64, r11 on x86-64
/// SysV and Win64), so arguments and the return address pass through untouched.
const Trampoline = if (!supported) struct {
    const stride = 16;
    fn emit(_: *CodeBuffer, _: usize) !void {
        return error.UnsupportedHost;
    }
} else switch (builtin.cpu.arch) {
    .aarch64 => struct {
        /// movz/movk x16 (up to 4) + `ldr x16, [x16]` + `br x16`, padded with `udf`.
        const stride = 24;
        fn emit(cb: *CodeBuffer, slot: usize) !void {
            const a = @import("../aarch64/assembler.zig").AssemblerT(.{}).init(cb);
            const end = cb.offset() + stride;
            try a.movImm(.x16, slot);
            try a.ldr(.x16, .x16, 0);
            try a.word(0xD61F0200); // br x16
            while (cb.offset() < end) try a.udf(0);
        }
    },
    .x86_64 => struct {
        /// `mov r11, imm64` (10 bytes) + `jmp qword ptr [r11]` (3 bytes), padded with `int3`.
        const stride = 16;
        fn emit(cb: *CodeBuffer, slot: usize) !void {
            const a = @import("../x86_64/assembler.zig").AssemblerT(.{}).init(cb);
            try a.movabs(.r11, slot);
            try cb.emitBytes(&.{ 0x41, 0xFF, 0x23, 0xCC, 0xCC, 0xCC });
        }
    },
    else => unreachable,
};

/// `JitRuntime.add` maps whole 16 KiB granules (its private `page` constant): fill one with
/// trampolines.
const page_bytes = 16384;
const per_page = page_bytes / Trampoline.stride;

/// A mapping of `per_page` trampolines and their slots (heap memory: code pages are W^X).
const Page = struct {
    code: JitRuntime.Function,
    slots: [per_page]std.atomic.Value(usize),
};

/// Where every slot points until its function is defined.
fn undefinedEntry() callconv(.c) noreturn {
    @panic("pyrlyn_jit.engine: called a function that was reserved but never defined");
}

/// Engine configuration.
pub const Options = struct {
    /// Calls before tier-up (0: compile optimized immediately, no profiling tier).
    hot_calls: u32 = 1000,
    /// Count blocks/edges in tier 0 (feeds inlining decisions and block layout). Without it,
    /// tier 0 counts only calls (for `auto_tier_up` and `poll`).
    profile: bool = true,
    /// Tier up from inside the call that reaches `hot_calls` (else only `tierUp`/`poll`).
    auto_tier_up: bool = true,
    /// Tier-1 optimizer options (`resolver` and `profile` are filled in by the engine; a
    /// `resolver` given here is consulted for call targets that are not engine functions).
    opt: opt.Options = .{ .level = .o2, .vector = opt.VectorTarget.host },
};

/// Handle of an engine function (index in reservation order).
pub const FuncId = enum(u32) { _ };

/// Code a function's entry point currently runs.
pub const Tier = enum { baseline, optimized };

/// Engine counters (`Engine.stats`).
pub const Stats = struct {
    /// Reserved functions.
    functions: u32 = 0,
    /// Host compiles that were installed (tier 0, tier 1 and fallbacks).
    compiles: u32 = 0,
    /// Functions switched to optimized code.
    tier_ups: u32 = 0,
    /// Call sites inlined by installed tier-1 code.
    inlined_calls: u32 = 0,
    /// Failed tier-ups (the function keeps its previous code).
    failed: u32 = 0,
    /// Bytes of generated function code (trampolines excluded).
    code_bytes: usize = 0,
    /// Executable memory mapped (function code and trampoline pages).
    mapped_bytes: usize = 0,
    /// Error of the latest failed tier-up.
    last_error: ?anyerror = null,
};

/// Per-function state; owned by the engine and only touched under the lock.
const Fn = struct {
    /// Engine copy (entry block prepended); null while only reserved.
    ir: ?ir.Func = null,
    /// `blocks ++ taken` (stored numbering) with `profile`, else one call counter. Tier-0 code
    /// writes it directly, without atomics.
    counters: []u64 = &.{},
    tier: Tier = .baseline,
    /// Latest tier-up failed; `poll` skips it until `tierUp` succeeds.
    failed: bool = false,
};

/// Tiered JIT engine. Create with `create`, release with `destroy`.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    options: Options,
    mutex: std.atomic.Mutex = .unlocked,
    pages: std.ArrayList(*Page) = .empty,
    funcs: std.ArrayList(Fn) = .empty,
    /// Every compiled function, released only by `destroy` (callers may still run old code).
    code: std.ArrayList(JitRuntime.Function) = .empty,
    st: Stats = .{},

    /// New engine. `gpa` must be thread safe if tier-ups can happen from several threads.
    pub fn create(gpa: std.mem.Allocator, options: Options) !*Engine {
        if (!supported) return error.UnsupportedHost;
        const e = try gpa.create(Engine);
        e.* = .{ .gpa = gpa, .options = options };
        return e;
    }

    /// Frees all code, trampolines and IR. No engine code may be running or called later.
    pub fn destroy(e: *Engine) void {
        const gpa = e.gpa;
        for (e.code.items) |c| c.release();
        e.code.deinit(gpa);
        for (e.pages.items) |p| {
            p.code.release();
            gpa.destroy(p);
        }
        e.pages.deinit(gpa);
        for (e.funcs.items) |*fd| if (fd.ir) |*g| {
            g.deinit();
            gpa.free(fd.counters);
        };
        e.funcs.deinit(gpa);
        gpa.destroy(e);
    }

    /// Reserves a function id and its entry point (`address`) before its IR exists, e.g. for
    /// recursion or mutual calls. Calling it before `define` panics.
    pub fn reserve(e: *Engine) !FuncId {
        e.lock();
        defer e.unlock();
        const i = e.funcs.items.len;
        if (i == std.math.maxInt(u32)) return error.TooManyFunctions;
        if (i / per_page == e.pages.items.len) try e.newPage();
        try e.funcs.append(e.gpa, .{});
        e.st.functions += 1;
        return @fromBackingInt(@intCast(i));
    }

    /// Gives a reserved function its IR (deep-copied; `f` is only read) and compiles tier 0,
    /// or tier 1 directly when `hot_calls == 0` (falling back to unoptimized code on failure).
    /// `f` must pass `verify` and `typeCheck`; its parameters and result are i64 (C ABI).
    pub fn define(e: *Engine, id: FuncId, f: *const ir.Func) !void {
        var fd = try e.prepare(f);
        e.lock();
        defer e.unlock();
        const i = @backingInt(id);
        if (i >= e.funcs.items.len or e.funcs.items[i].ir != null) {
            fd.ir.?.deinit();
            e.gpa.free(fd.counters);
            return if (i >= e.funcs.items.len) error.UnknownFunction else error.AlreadyDefined;
        }
        e.funcs.items[i] = fd;
        e.install(i) catch |err| {
            e.funcs.items[i].ir.?.deinit();
            e.gpa.free(fd.counters);
            e.funcs.items[i] = .{};
            return err;
        };
    }

    /// `reserve` + `define`. A failed `define` leaves the reserved id undefined.
    pub fn add(e: *Engine, f: *const ir.Func) !FuncId {
        const id = try e.reserve();
        try e.define(id, f);
        return id;
    }

    /// Stable entry point of `id` (use it as the `call` target in IR or as a C function).
    pub fn address(e: *Engine, id: FuncId) usize {
        e.lock();
        defer e.unlock();
        const i = @backingInt(id);
        std.debug.assert(i < e.funcs.items.len);
        return @intFromPtr(e.pages.items[i / per_page].code.mem.ptr) + i % per_page * Trampoline.stride;
    }

    /// `address(id)` as a function pointer, e.g. `*const fn (i64) callconv(.c) i64`.
    pub fn ptr(e: *Engine, id: FuncId, comptime F: type) F {
        return @ptrFromInt(e.address(id));
    }

    /// Optimizes `id` now (with the profile so far and inlining of engine callees). Clears a
    /// recorded failure first; on error the function keeps its current code. No-op when
    /// already optimized.
    pub fn tierUp(e: *Engine, id: FuncId) !void {
        e.lock();
        defer e.unlock();
        const i = @backingInt(id);
        if (i >= e.funcs.items.len) return error.UnknownFunction;
        e.funcs.items[i].failed = false;
        e.tierUpLocked(i) catch |err| {
            e.fail(i, err);
            return err;
        };
    }

    /// Tiers up every defined baseline function called `hot_calls` times or more that has not
    /// failed (catches up on hooks that found the lock busy or that `auto_tier_up` disabled).
    /// Failures are recorded in `stats`. Returns the number of functions tiered up.
    pub fn poll(e: *Engine) u32 {
        e.lock();
        defer e.unlock();
        var n: u32 = 0;
        for (e.funcs.items, 0..) |*fd, i| {
            if (fd.ir == null or fd.tier != .baseline or fd.failed) continue;
            if (@atomicLoad(u64, &fd.counters[0], .monotonic) < e.options.hot_calls) continue;
            if (e.tierUpLocked(@intCast(i))) |_| {
                n += 1;
            } else |err| e.fail(@intCast(i), err);
        }
        return n;
    }

    /// Current tier of `id`.
    pub fn tier(e: *Engine, id: FuncId) Tier {
        e.lock();
        defer e.unlock();
        return e.funcs.items[@backingInt(id)].tier;
    }

    /// Calls of `id` counted by its tier-0 code (counting stops after tier-up).
    pub fn calls(e: *Engine, id: FuncId) u64 {
        e.lock();
        defer e.unlock();
        const c = e.funcs.items[@backingInt(id)].counters;
        return if (c.len == 0) 0 else @atomicLoad(u64, &c[0], .monotonic);
    }

    /// Live tier-0 counters of `id` indexed by the block numbers of the IR given to `define`
    /// (empty without `Options.profile` or before `define`). Valid until `destroy`.
    pub fn profile(e: *Engine, id: FuncId) opt.Profile {
        e.lock();
        defer e.unlock();
        const fd = &e.funcs.items[@backingInt(id)];
        if (!e.options.profile or fd.ir == null) return .{};
        const n = fd.counters.len / 2;
        return .{ .blocks = fd.counters[1..n], .taken = fd.counters[n + 1 ..] };
    }

    /// Snapshot of the engine counters.
    pub fn stats(e: *Engine) Stats {
        e.lock();
        defer e.unlock();
        return e.st;
    }

    fn lock(e: *Engine) void {
        while (!e.mutex.tryLock()) std.Thread.yield() catch std.atomic.spinLoopHint();
    }
    fn unlock(e: *Engine) void {
        e.mutex.unlock();
    }

    fn slot(e: *Engine, i: u32) *std.atomic.Value(usize) {
        return &e.pages.items[i / per_page].slots[i % per_page];
    }

    /// Maps a trampoline address to its function index.
    fn indexOf(e: *Engine, addr: usize) ?u32 {
        for (e.pages.items, 0..) |p, k| {
            const base = @intFromPtr(p.code.mem.ptr);
            if (addr < base or addr - base >= per_page * Trampoline.stride) continue;
            if ((addr - base) % Trampoline.stride != 0) return null;
            const i = k * per_page + (addr - base) / Trampoline.stride;
            return if (i < e.funcs.items.len) @intCast(i) else null;
        }
        return null;
    }

    /// Maps and fills the next trampoline page; its slots start at `undefinedEntry`.
    fn newPage(e: *Engine) !void {
        const p = try e.gpa.create(Page);
        errdefer e.gpa.destroy(p);
        for (&p.slots) |*s| s.* = .init(@intFromPtr(&undefinedEntry));
        var cb = CodeBuffer.init(e.gpa);
        defer cb.deinit();
        for (&p.slots) |*s| try Trampoline.emit(&cb, @intFromPtr(s));
        try cb.finalize();
        try e.pages.ensureUnusedCapacity(e.gpa, 1);
        p.code = try JitRuntime.add(&cb);
        e.pages.appendAssumeCapacity(p);
        e.st.mapped_bytes += p.code.mem.len;
    }

    /// Validated engine copy of `f` with its counters (zeroed).
    fn prepare(e: *Engine, f: *const ir.Func) !Fn {
        try f.verify();
        e.gpa.free(try f.typeCheck(e.gpa));
        var g = try withEntry(e.gpa, f);
        errdefer g.deinit();
        const counters = try e.gpa.alloc(u64, if (e.options.profile) 2 * g.blocks.items.len else 1);
        @memset(counters, 0);
        return .{ .ir = g, .counters = counters };
    }

    /// First code of a freshly defined function.
    fn install(e: *Engine, i: u32) !void {
        if (e.options.hot_calls == 0) {
            e.tierUpLocked(i) catch |err| {
                e.fail(i, err);
                e.slot(i).store(try e.compile(&e.funcs.items[i].ir.?), .release);
            };
            return;
        }
        const fd = &e.funcs.items[i];
        const n = fd.ir.?.blocks.items.len;
        const counters: profiling.Counters = if (e.options.profile)
            .{ .blocks = fd.counters[0..n], .taken = fd.counters[n..] }
        else
            .{ .blocks = fd.counters, .taken = fd.counters[0..0] };
        const hook: ?profiling.Hook = if (e.options.auto_tier_up) .{
            .at = e.options.hot_calls,
            .addr = @intFromPtr(&hookEntry),
            .arg0 = @bitCast(@intFromPtr(e)),
            .arg1 = i,
        } else null;
        var g = try profiling.instrument(e.gpa, &fd.ir.?, counters, hook);
        defer g.deinit();
        e.slot(i).store(try e.compile(&g), .release);
    }

    fn fail(e: *Engine, i: u32, err: anyerror) void {
        e.funcs.items[i].failed = true;
        e.st.failed += 1;
        e.st.last_error = err;
    }

    /// Optimizes and installs tier 1 for defined function `i` (no-op when optimized). Vector
    /// IR the host compiler rejects is retried without vectorization.
    fn tierUpLocked(e: *Engine, i: u32) !void {
        const fd = &e.funcs.items[i];
        const base = if (fd.ir) |*g| g else return error.UndefinedFunction;
        if (fd.tier == .optimized) return;
        var o = e.options.opt;
        o.resolver = .{ .ctx = e, .lookup = lookup };
        // Copy the counters: tier-0 code may keep updating them while the optimizer reads.
        const counts: ?[]u64 = if (e.options.profile) try e.gpa.alloc(u64, fd.counters.len) else null;
        defer if (counts) |c| e.gpa.free(c);
        var prof: opt.Profile = .{};
        if (counts) |c| {
            for (c, fd.counters) |*d, *s| d.* = @atomicLoad(u64, s, .monotonic);
            prof = .{ .blocks = c[0 .. c.len / 2], .taken = c[c.len / 2 ..] };
            o.profile = &prof;
        } else o.profile = null;
        const entry = e.optimized(base, o) catch |err| retry: {
            if (!o.vector.enabled()) return err;
            o.vector = .none;
            break :retry try e.optimized(base, o);
        };
        fd.tier = .optimized;
        e.st.tier_ups += 1;
        e.slot(i).store(entry, .release);
    }

    /// Optimized copy of `base`, compiled; returns its entry address.
    fn optimized(e: *Engine, base: *const ir.Func, o: opt.Options) !usize {
        var g = try profiling.clone(e.gpa, base);
        defer g.deinit();
        const os = try opt.optimize(e.gpa, &g, o);
        const entry = try e.compile(&g);
        e.st.inlined_calls += os.inlined_calls;
        return entry;
    }

    /// Host-compiles `f` into new executable memory owned by the engine; returns its address.
    fn compile(e: *Engine, f: *const ir.Func) !usize {
        if (!supported) return error.UnsupportedHost;
        var cb = CodeBuffer.init(e.gpa);
        defer cb.deinit();
        _ = try host.compile(e.gpa, f, &cb, .{});
        try e.code.ensureUnusedCapacity(e.gpa, 1);
        const c = try JitRuntime.add(&cb);
        e.code.appendAssumeCapacity(c);
        e.st.compiles += 1;
        e.st.code_bytes += cb.code().len;
        e.st.mapped_bytes += c.mem.len;
        return @intFromPtr(c.mem.ptr);
    }

    /// `opt.Resolver.lookup`: engine trampolines resolve to the callee's engine IR (called
    /// under the lock, so the pointer stays valid for the whole `optimize`); other addresses go
    /// to the resolver in `Options.opt`, if any.
    fn lookup(ctx: ?*anyopaque, addr: usize) ?*const ir.Func {
        const e: *Engine = @ptrCast(@alignCast(ctx.?));
        if (e.indexOf(addr)) |i| if (e.funcs.items[i].ir) |*g| return g;
        const user = e.options.opt.resolver orelse return null;
        return user.lookup(user.ctx, addr);
    }

    /// Tier-0 hook (`profile.Hook`): tiers function `id` up from inside its `hot_calls`-th
    /// call. Returns at once when another compile holds the lock (`poll` catches up); errors
    /// are recorded in `stats`. The running call finishes in tier 0.
    fn hookEntry(engine: i64, id: i64) callconv(.c) i64 {
        const e: *Engine = @ptrFromInt(@as(usize, @bitCast(engine)));
        if (!e.mutex.tryLock()) return 0;
        defer e.unlock();
        const i: u32 = @intCast(id);
        const fd = &e.funcs.items[i];
        if (fd.tier == .baseline and !fd.failed) e.tierUpLocked(i) catch |err| e.fail(i, err);
        return 0;
    }
};

/// Deep copy of `f` with a new entry block 0 that branches to the old block 0 (passing its
/// parameters to themselves: block-0 parameters are not assigned on function entry), so no
/// branch targets the entry block. Every old block index moves up by one.
fn withEntry(gpa: std.mem.Allocator, f: *const ir.Func) !ir.Func {
    var g = try profiling.clone(gpa, f);
    errdefer g.deinit();
    for (g.blocks.items) |*b| switch (b.ops.items[b.ops.items.len - 1]) {
        .br => |*tg| tg.* += 1,
        .bra => |*x| x.target += 1,
        inline .cbr, .cbra => |*x| {
            x.then += 1;
            x.@"else" += 1;
        },
        else => {},
    };
    const a = g.alloc();
    const params = g.blocks.items[0].params;
    var ops = try std.ArrayList(ir.Op).initCapacity(a, 1);
    ops.appendAssumeCapacity(if (params.len == 0) .{ .br = 1 } else .{ .bra = .{ .target = 1, .args = params } });
    try g.blocks.insert(a, 0, .{ .name = "engine.entry", .ops = ops });
    return g;
}

const t = std.testing;

const Fn1 = *const fn (i64) callconv(.c) i64;

fn parseFmt(comptime src: []const u8, args: anytype) !ir.Func {
    const text = try std.fmt.allocPrint(t.allocator, src, args);
    defer t.allocator.free(text);
    return ir.parse(t.allocator, text);
}

/// sum of i*i for i < n (loop header is block 1; `then` of block 1 is taken n times).
const squares_src =
    \\func squares 1
    \\e:
    \\  v1 = const 0
    \\  v2 = const 0
    \\  v3 = const 1
    \\  br h
    \\h:
    \\  cbr lt v2 v0 body done
    \\body:
    \\  v4 = mul v2 v2
    \\  v1 = add v1 v4
    \\  v2 = add v2 v3
    \\  br h
    \\done:
    \\  ret v1
;

fn squares(n: i64) i64 {
    var s: i64 = 0;
    var i: i64 = 0;
    while (i < n) : (i += 1) s +%= i *% i;
    return s;
}

test "engine: a loop function tiers up automatically after hot_calls calls" {
    if (!supported) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, squares_src);
    defer f.deinit();
    const e = try Engine.create(t.allocator, .{ .hot_calls = 5 });
    defer e.destroy();
    const id = try e.add(&f);
    const call = e.ptr(id, Fn1);
    for (0..12) |k| {
        const n: i64 = @as(i64, @intCast(k)) * 7 - 3;
        try t.expectEqual(try ir.interpret(&f, &.{n}, 1 << 20), call(n));
        try t.expectEqual(squares(n), call(n));
        // The hook fires inside the 5th call (k = 2, second call of that iteration).
        try t.expectEqual(@as(Tier, if (k >= 2) .optimized else .baseline), e.tier(id));
    }
    const s = e.stats();
    try t.expectEqual(@as(u32, 2), s.compiles);
    try t.expectEqual(@as(u32, 1), s.tier_ups);
    try t.expectEqual(@as(u32, 0), s.failed);
    try t.expectEqual(@as(u64, 5), e.calls(id)); // counting stops with tier 0
}

test "engine: calls through address() are inlined at tier-up" {
    if (!supported) return error.SkipZigTest;
    const e = try Engine.create(t.allocator, .{ .hot_calls = 3 });
    defer e.destroy();
    var sq = try ir.parse(t.allocator,
        \\func sq 1
        \\e:
        \\  v1 = mul v0 v0
        \\  v2 = const 7
        \\  v3 = xor v1 v2
        \\  ret v3
    );
    defer sq.deinit();
    const callee = try e.add(&sq);
    var caller = try parseFmt(
        \\func caller 1
        \\e:
        \\  v1 = const 0
        \\  v2 = const 0
        \\  v3 = const 1
        \\  br h
        \\h:
        \\  cbr lt v2 v0 body done
        \\body:
        \\  v4 = call {d} v2
        \\  v1 = add v1 v4
        \\  v2 = add v2 v3
        \\  br h
        \\done:
        \\  ret v1
    , .{e.address(callee)});
    defer caller.deinit();
    const id = try e.add(&caller);
    const call = e.ptr(id, Fn1);
    for (0..8) |k| {
        const n: i64 = @intCast(k * 5);
        // The interpreter calls the callee through its trampoline too.
        try t.expectEqual(try ir.interpret(&caller, &.{n}, 1 << 20), call(n));
    }
    try t.expectEqual(Tier.optimized, e.tier(id));
    try t.expect(e.stats().inlined_calls >= 1);
    try t.expectEqual(@as(i64, 7), e.ptr(callee, Fn1)(0));
}

fn fib(n: i64) i64 {
    return if (n < 2) n else fib(n - 1) + fib(n - 2);
}

test "engine: a recursive function calls itself through its own address" {
    if (!supported) return error.SkipZigTest;
    const e = try Engine.create(t.allocator, .{ .hot_calls = 20 });
    defer e.destroy();
    const id = try e.reserve();
    var f = try parseFmt(
        \\func fib 1
        \\e:
        \\  v1 = const 2
        \\  cbr lt v0 v1 base rec
        \\base:
        \\  ret v0
        \\rec:
        \\  v2 = const 1
        \\  v3 = sub v0 v2
        \\  v4 = call {d} v3
        \\  v5 = sub v3 v2
        \\  v6 = call {d} v5
        \\  v7 = add v4 v6
        \\  ret v7
    , .{ e.address(id), e.address(id) });
    defer f.deinit();
    try e.define(id, &f);
    try t.expectError(error.AlreadyDefined, e.define(id, &f));
    const call = e.ptr(id, Fn1);
    try t.expectEqual(@as(i64, 3), call(4)); // 9 calls: stays in tier 0
    try t.expectEqual(Tier.baseline, e.tier(id));
    for (0..16) |n| try t.expectEqual(fib(@intCast(n)), call(@intCast(n)));
    try t.expectEqual(Tier.optimized, e.tier(id));
    try t.expect(e.stats().inlined_calls >= 1);
    try t.expectEqual(fib(20), call(20));
    try t.expectEqual(try ir.interpret(&f, &.{12}, 1 << 20), call(12));
}

test "engine: profile counts of a loop" {
    if (!supported) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, squares_src);
    defer f.deinit();
    const e = try Engine.create(t.allocator, .{ .hot_calls = 100, .auto_tier_up = false });
    defer e.destroy();
    const id = try e.add(&f);
    const call = e.ptr(id, Fn1);
    try t.expectEqual(squares(3), call(3));
    try t.expectEqual(squares(5), call(5));
    try t.expectEqual(@as(i64, 0), call(-1));
    const p = e.profile(id);
    try t.expectEqualSlices(u64, &.{ 3, 4 + 6 + 1, 3 + 5, 3 }, p.blocks);
    try t.expectEqual(@as(u64, 8), p.taken[1]);
    try t.expectEqual(@as(u64, 3), e.calls(id));
    try t.expectEqual(Tier.baseline, e.tier(id));
}

test "engine: poll, explicit tierUp, hot_calls = 0 and profile = false" {
    if (!supported) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, squares_src);
    defer f.deinit();
    {
        const e = try Engine.create(t.allocator, .{ .hot_calls = 3, .auto_tier_up = false });
        defer e.destroy();
        const a = try e.add(&f);
        const b = try e.add(&f);
        try t.expectEqual(squares(9), e.ptr(a, Fn1)(9));
        try t.expectEqual(squares(9), e.ptr(a, Fn1)(9));
        try t.expectEqual(@as(u32, 0), e.poll());
        try t.expectEqual(squares(9), e.ptr(a, Fn1)(9));
        try t.expectEqual(@as(u32, 1), e.poll());
        try t.expectEqual(Tier.optimized, e.tier(a));
        try t.expectEqual(Tier.baseline, e.tier(b));
        try e.tierUp(b);
        try e.tierUp(b); // no-op
        try t.expectEqual(Tier.optimized, e.tier(b));
        try t.expectEqual(@as(u32, 0), e.poll());
        for ([_]FuncId{ a, b }) |id| try t.expectEqual(squares(30), e.ptr(id, Fn1)(30));
        try t.expectEqual(@as(u32, 4), e.stats().compiles);
    }
    {
        const e = try Engine.create(t.allocator, .{ .hot_calls = 0 });
        defer e.destroy();
        const id = try e.add(&f);
        try t.expectEqual(Tier.optimized, e.tier(id));
        try t.expectEqual(squares(11), e.ptr(id, Fn1)(11));
        try t.expectEqual(@as(u32, 1), e.stats().compiles);
        try t.expectEqual(@as(u64, 0), e.calls(id));
    }
    {
        const e = try Engine.create(t.allocator, .{ .hot_calls = 4, .profile = false });
        defer e.destroy();
        const id = try e.add(&f);
        for (0..3) |_| try t.expectEqual(squares(6), e.ptr(id, Fn1)(6));
        try t.expectEqual(@as(usize, 0), e.profile(id).blocks.len);
        try t.expectEqual(@as(u64, 3), e.calls(id));
        try t.expectEqual(squares(6), e.ptr(id, Fn1)(6));
        try t.expectEqual(Tier.optimized, e.tier(id));
    }
}

test "engine: a failed tier-up keeps tier 0 and is recorded" {
    if (!supported) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, squares_src);
    defer f.deinit();
    var fa = std.testing.FailingAllocator.init(t.allocator, .{});
    const e = try Engine.create(fa.allocator(), .{ .hot_calls = 2, .auto_tier_up = false });
    defer e.destroy();
    const id = try e.add(&f);
    for (0..3) |_| try t.expectEqual(squares(10), e.ptr(id, Fn1)(10));
    fa.fail_index = fa.alloc_index; // every later allocation fails
    try t.expectError(error.OutOfMemory, e.tierUp(id));
    try t.expectEqual(@as(u32, 0), e.poll()); // marked failed: skipped
    fa.fail_index = std.math.maxInt(usize);
    var s = e.stats();
    try t.expectEqual(@as(u32, 1), s.failed);
    try t.expectEqual(@as(?anyerror, error.OutOfMemory), s.last_error);
    try t.expectEqual(Tier.baseline, e.tier(id));
    try t.expectEqual(squares(10), e.ptr(id, Fn1)(10));
    try e.tierUp(id);
    s = e.stats();
    try t.expectEqual(Tier.optimized, e.tier(id));
    try t.expectEqual(@as(u32, 1), s.tier_ups);
    try t.expectEqual(squares(10), e.ptr(id, Fn1)(10));
}

test "engine: threads call a function while it tiers up" {
    if (!supported or builtin.single_threaded) return error.SkipZigTest;
    var f = try ir.parse(t.allocator, squares_src);
    defer f.deinit();
    const e = try Engine.create(t.allocator, .{ .hot_calls = 200 });
    defer e.destroy();
    const id = try e.add(&f);
    const Worker = struct {
        fn run(call: Fn1, seed: usize, bad: *std.atomic.Value(u32)) void {
            for (0..3000) |k| {
                const n: i64 = @intCast((k * 31 + seed * 17) % 97);
                if (call(n) != squares(n)) _ = bad.fetchAdd(1, .monotonic);
            }
        }
    };
    var bad: std.atomic.Value(u32) = .init(0);
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*th, s| th.* = try std.Thread.spawn(.{}, Worker.run, .{ e.ptr(id, Fn1), s, &bad });
    for (threads) |th| th.join();
    try t.expectEqual(@as(u32, 0), bad.load(.monotonic));
    // The hook may have found the lock busy or seen a lost count: poll catches up.
    _ = e.poll();
    try t.expectEqual(Tier.optimized, e.tier(id));
    try t.expectEqual(@as(u32, 1), e.stats().tier_ups);
    try t.expectEqual(squares(96), e.ptr(id, Fn1)(96));
}
