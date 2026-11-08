//! Executable memory with a bounded cache of released single-page mappings.
//! macOS/arm64: MAP_JIT + pthread_jit_write_protect_np + sys_icache_invalidate.
//! Other POSIX hosts: mmap RW, copy, mprotect RX (W^X, never RWX).
//! Windows: VirtualAlloc RW, copy, VirtualProtect RX (W^X, never RWX).
const std = @import("std");
const builtin = @import("builtin");
const CodeBuffer = @import("code_buffer.zig").CodeBuffer;

extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn mprotect(addr: *anyopaque, len: usize, prot: c_int) c_int;
extern "c" fn pthread_jit_write_protect_np(enabled: c_int) void;
extern "c" fn sys_icache_invalidate(start: *anyopaque, len: usize) void;
extern "c" fn __clear_cache(start: *anyopaque, end: *anyopaque) void;

const is_windows = builtin.os.tag == .windows;
const MEM_COMMIT = 0x1000;
const MEM_RESERVE = 0x2000;
const MEM_RELEASE = 0x8000;
const PAGE_READWRITE = 0x04;
const PAGE_EXECUTE_READ = 0x20;
extern "kernel32" fn VirtualAlloc(addr: ?*anyopaque, size: usize, kind: u32, protect: u32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn VirtualFree(addr: *anyopaque, size: usize, kind: u32) callconv(.winapi) i32;
extern "kernel32" fn VirtualProtect(addr: *anyopaque, size: usize, protect: u32, old: *u32) callconv(.winapi) i32;
extern "kernel32" fn FlushInstructionCache(process: *anyopaque, addr: ?*const anyopaque, size: usize) callconv(.winapi) i32;
extern "kernel32" fn GetCurrentProcess() callconv(.winapi) *anyopaque;

fn unmap(p: *anyopaque, len: usize) void {
    if (is_windows) _ = VirtualFree(p, 0, MEM_RELEASE) else _ = munmap(p, len);
}

const PROT_READ = 1;
const PROT_WRITE = 2;
const PROT_EXEC = 4;
const is_darwin = builtin.os.tag.isDarwin();
const MAP_PRIVATE = 0x02;
const MAP_ANON: c_int = if (is_darwin) 0x1000 else 0x20;
const MAP_JIT = 0x800;
/// Architectures whose instruction cache is not coherent with data writes.
const needs_icache_flush = switch (builtin.cpu.arch) {
    .aarch64, .riscv64, .riscv32 => true,
    else => false,
};
const use_map_jit = is_darwin and builtin.cpu.arch == .aarch64;
const MAP_FAILED: usize = std.math.maxInt(usize);

pub const Error = error{ MapFailed, ProtectFailed, OutOfBounds };

pub const Function = struct {
    mem: []align(4096) u8,
    /// Initialized code prefix; the rest is trap-filled. Unknown for manually constructed
    /// handles, so their whole mapping is treated as dirty when recycled.
    code_len: usize = std.math.maxInt(usize),
    pub fn ptr(self: Function, comptime F: type) F {
        return @ptrCast(@alignCast(self.mem.ptr));
    }
    /// Releases ownership. Do not access the function afterward; the mapping may be reused.
    pub fn release(self: Function) void {
        if (!cacheReleased(self)) unmap(self.mem.ptr, self.mem.len);
    }
};

/// Allocation granule: 16 KiB covers both 4K (x86) and 16K (Apple arm64) pages.
const page = 16384;

// Only released mappings enter the cache. Each live Function retains an exclusive mapping;
// no executable page is ever shared between independently owned functions. Contention falls
// back to the OS rather than blocking. At most 8 * 16 KiB are retained per linked runtime.
const cache_capacity = 8;
var cache_mutex: std.atomic.Mutex = .unlocked;
var cache_count: usize = 0;
const Cached = struct { mem: []align(4096) u8, used: usize };
var cache: [cache_capacity]Cached = undefined;

fn takeCached(size: usize) ?Cached {
    if (size != page or !cache_mutex.tryLock()) return null;
    defer cache_mutex.unlock();
    if (cache_count == 0) return null;
    cache_count -= 1;
    return cache[cache_count];
}

fn cacheReleased(f: Function) bool {
    const mem = f.mem;
    if (mem.len != page or !cache_mutex.tryLock()) return false;
    defer cache_mutex.unlock();
    if (cache_count == cache_capacity) return false;
    cache[cache_count] = .{ .mem = mem, .used = @min(f.code_len, mem.len) };
    cache_count += 1;
    return true;
}

/// Unmaps idle cached pages; live functions are unaffected. Returns bytes released.
/// Best effort: returns zero under contention. At most 128 KiB are cached per runtime.
pub fn trimCache() usize {
    if (!cache_mutex.tryLock()) return 0;
    const n = cache_count;
    var idle: [cache_capacity]Cached = undefined;
    @memcpy(idle[0..n], cache[0..n]);
    cache_count = 0;
    cache_mutex.unlock();
    for (idle[0..n]) |entry| unmap(entry.mem.ptr, entry.mem.len);
    return n * page;
}

/// Copies finalized code into an exclusively owned executable mapping. Released single-page
/// mappings may be recycled; `Function.release` transfers ownership back to the runtime.
pub fn add(cb: *const CodeBuffer) Error!Function {
    const code = cb.code();
    const size = std.mem.alignForward(usize, @max(code.len, 1), page);
    const recycled = takeCached(size);
    const mem = if (recycled) |entry| entry.mem else try mapWritable(size);
    // The untouched tail of a recycled mapping is already trap-filled and icache-clean.
    // Rewrite/flush both the new function and any old code extending beyond it.
    const dirty_len = if (recycled) |entry| @max(code.len, entry.used) else mem.len;
    errdefer unmap(mem.ptr, mem.len);
    if (recycled != null and !use_map_jit) try makeWritable(mem);
    if (use_map_jit) {
        // Each cache entry is exclusively owned before opening the thread-local write window.
        pthread_jit_write_protect_np(0);
        defer pthread_jit_write_protect_np(1);
        @memcpy(mem[0..code.len], code);
        fillTrap(mem[code.len..dirty_len]);
    } else {
        @memcpy(mem[0..code.len], code);
        fillTrap(mem[code.len..dirty_len]);
        try makeExecutable(mem);
    }
    // Include overwritten old code in the flush, even when the new function is shorter.
    // Darwin x86 also needs this for Rosetta's translation cache.
    if (is_windows) {
        _ = FlushInstructionCache(GetCurrentProcess(), mem.ptr, dirty_len);
    } else if (is_darwin) {
        sys_icache_invalidate(mem.ptr, dirty_len);
    } else if (needs_icache_flush) {
        __clear_cache(mem.ptr, mem.ptr + dirty_len);
    }
    return .{ .mem = mem, .code_len = code.len };
}

fn mapWritable(size: usize) Error![]align(4096) u8 {
    const p = if (is_windows)
        VirtualAlloc(null, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) orelse return error.MapFailed
    else blk: {
        const flags: c_int = MAP_PRIVATE | MAP_ANON | (if (use_map_jit) MAP_JIT else 0);
        const prot: c_int = if (use_map_jit) PROT_READ | PROT_WRITE | PROT_EXEC else PROT_READ | PROT_WRITE;
        const ptr = mmap(null, size, prot, flags, -1, 0) orelse return error.MapFailed;
        if (@intFromPtr(ptr) == MAP_FAILED) return error.MapFailed;
        break :blk ptr;
    };
    return asSlice(p, size);
}

fn makeWritable(mem: []align(4096) u8) Error!void {
    if (is_windows) {
        var old: u32 = 0;
        if (VirtualProtect(mem.ptr, mem.len, PAGE_READWRITE, &old) == 0) return error.ProtectFailed;
    } else if (mprotect(mem.ptr, mem.len, PROT_READ | PROT_WRITE) != 0) return error.ProtectFailed;
}

fn makeExecutable(mem: []align(4096) u8) Error!void {
    if (is_windows) {
        var old: u32 = 0;
        if (VirtualProtect(mem.ptr, mem.len, PAGE_EXECUTE_READ, &old) == 0) return error.ProtectFailed;
    } else if (mprotect(mem.ptr, mem.len, PROT_READ | PROT_EXEC) != 0) return error.ProtectFailed;
}

/// Fills unused executable memory with traps: `int3` (0xCC) on x86-64, `udf #0` (zero words) on AArch64.
pub fn fillTrap(mem: []u8) void {
    @memset(mem, if (builtin.cpu.arch == .x86_64) 0xCC else 0x00);
}

// ---- Dual mapping (AsmJit: VirtMem::allocDualMapping) -------------------------------------
// One physical region, two views: `rw` is never executable, `rx` is never writable. Code can be
// patched through `rw` while `rx` keeps running; no thread-local W^X toggling is needed.
// macOS: anonymous RW mapping + mach_vm_remap (shared, copy=false) + mprotect RX on the alias.
// Linux: memfd_create + two MAP_SHARED mappings of the same fd.

extern "c" fn mach_task_self() c_uint;
extern "c" fn mach_vm_remap(target: c_uint, addr: *u64, size: u64, mask: u64, flags: c_int, src_task: c_uint, src: u64, copy: c_int, cur: *c_int, max: *c_int, inherit: c_uint) c_int;
extern "c" fn memfd_create(name: [*:0]const u8, flags: c_uint) c_int;
extern "c" fn ftruncate(fd: c_int, len: i64) c_int;
extern "c" fn close(fd: c_int) c_int;

const MAP_SHARED = 0x01;
const VM_FLAGS_ANYWHERE = 0x1;
const VM_INHERIT_DEFAULT = 1;

pub const DualFunction = struct {
    rx: []align(4096) u8,
    rw: []align(4096) u8,
    pub fn ptr(self: DualFunction, comptime F: type) F {
        return @ptrCast(@alignCast(self.rx.ptr));
    }
    /// Patches code through the RW view and flushes the icache for the RX view.
    /// Returns `OutOfBounds` (and writes nothing) if the range does not fit the mapping.
    pub fn write(self: DualFunction, offset: usize, bytes: []const u8) Error!void {
        if (offset > self.rw.len or bytes.len > self.rw.len - offset) return error.OutOfBounds;
        @memcpy(self.rw[offset..][0..bytes.len], bytes);
        flush(self.rx[offset..].ptr, bytes.len);
    }
    pub fn release(self: DualFunction) void {
        _ = munmap(self.rx.ptr, self.rx.len);
        _ = munmap(self.rw.ptr, self.rw.len);
    }
};

extern "c" fn sysctlbyname(name: [*:0]const u8, old: ?*anyopaque, oldlen: ?*usize, new: ?*anyopaque, newlen: usize) c_int;

/// True when running x86-64 code under Rosetta 2. Rosetta caches translations and does not see
/// writes made through the RW alias, so `DualFunction.write` patches are not reliably observed.
pub fn underRosetta() bool {
    if (!is_darwin or builtin.cpu.arch != .x86_64) return false;
    var v: c_int = 0;
    var n: usize = @sizeOf(c_int);
    return sysctlbyname("sysctl.proc_translated", &v, &n, null, 0) == 0 and v == 1;
}

pub const dual_mapping_supported = is_darwin or builtin.os.tag == .linux;

fn flush(p: [*]u8, len: usize) void {
    if (!needs_icache_flush) return;
    if (is_darwin) sys_icache_invalidate(p, len) else __clear_cache(p, p + len);
}

fn asSlice(p: *anyopaque, size: usize) []align(4096) u8 {
    return @as([*]align(4096) u8, @ptrCast(@alignCast(p)))[0..size];
}

/// Allocates a dual mapping of at least `min_size` bytes, trap-filled.
pub fn allocDual(min_size: usize) Error!DualFunction {
    if (!dual_mapping_supported) @compileError("dual mapping: unsupported OS");
    const size = std.mem.alignForward(usize, @max(min_size, 1), page);
    var rw: []align(4096) u8 = undefined;
    var rx: []align(4096) u8 = undefined;
    if (is_darwin) {
        const p = mmap(null, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0) orelse return error.MapFailed;
        if (@intFromPtr(p) == MAP_FAILED) return error.MapFailed;
        rw = asSlice(p, size);
        var addr: u64 = 0;
        var cur: c_int = 0;
        var max: c_int = 0;
        const t = mach_task_self();
        if (mach_vm_remap(t, &addr, size, 0, VM_FLAGS_ANYWHERE, t, @intFromPtr(p), 0, &cur, &max, VM_INHERIT_DEFAULT) != 0) {
            _ = munmap(p, size);
            return error.MapFailed;
        }
        rx = asSlice(@ptrFromInt(addr), size);
    } else {
        const fd = memfd_create("pyrlyn_jit", 1); // MFD_CLOEXEC
        if (fd < 0) return error.MapFailed;
        defer _ = close(fd);
        if (ftruncate(fd, @intCast(size)) != 0) return error.MapFailed;
        const a = mmap(null, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) orelse return error.MapFailed;
        if (@intFromPtr(a) == MAP_FAILED) return error.MapFailed;
        const b = mmap(null, size, PROT_READ, MAP_SHARED, fd, 0) orelse return error.MapFailed;
        if (@intFromPtr(b) == MAP_FAILED) {
            _ = munmap(a, size);
            return error.MapFailed;
        }
        rw = asSlice(a, size);
        rx = asSlice(b, size);
    }
    fillTrap(rw);
    if (mprotect(rx.ptr, size, PROT_READ | PROT_EXEC) != 0) {
        _ = munmap(rx.ptr, size);
        _ = munmap(rw.ptr, size);
        return error.ProtectFailed;
    }
    return .{ .rx = rx, .rw = rw };
}

/// Like `add`, but returns a dual mapping so the code can be patched later via `write`.
pub fn addDual(cb: *const CodeBuffer) Error!DualFunction {
    const code = cb.code();
    const f = try allocDual(code.len);
    f.write(0, code) catch unreachable; // allocDual sized the mapping for `code`
    return f;
}

test "runtime cache: bound, exclusive ownership, trap tail and trim" {
    const t = std.testing;
    _ = trimCache();
    defer _ = trimCache();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try cb.emitBytes(&@as([512]u8, @splat(0x42)));
    var functions: [cache_capacity + 2]Function = undefined;
    var live: usize = 0;
    defer for (functions[0..live]) |f| f.release();
    for (&functions) |*f| {
        f.* = try add(&cb);
        for (functions[0..live]) |other| try t.expect(f.mem.ptr != other.mem.ptr);
        live += 1;
    }
    const last_cached = functions[cache_capacity - 1].mem.ptr;
    for (functions[0..live]) |f| f.release();
    live = 0;
    cb.reset();
    try cb.emitBytes(&.{0x37});
    const short = try add(&cb);
    defer short.release();
    try t.expectEqual(last_cached, short.mem.ptr);
    try t.expectEqual(@as(u8, 0x37), short.mem[0]);
    const trap: u8 = if (builtin.cpu.arch == .x86_64) 0xCC else 0;
    for (short.mem[1..]) |byte| try t.expectEqual(trap, byte);
    try t.expectEqual((cache_capacity - 1) * page, trimCache());
    // Trimming the idle cache must not release the live mapping.
    try t.expectEqual(@as(u8, 0x37), short.mem[0]);
    try t.expectEqual(@as(usize, 0), trimCache());
}

test "runtime cache: large mappings and contention bypass retention" {
    const t = std.testing;
    _ = trimCache();
    defer _ = trimCache();
    var cb = CodeBuffer.init(t.allocator);
    defer cb.deinit();
    try cb.emitBytes(&@as([page + 1]u8, @splat(0)));
    const large = try add(&cb);
    try t.expectEqual(2 * page, large.mem.len);
    large.release();
    try t.expectEqual(@as(usize, 0), trimCache());
    cb.reset();
    try cb.emitBytes(&.{0});
    const small = try add(&cb);
    try t.expect(cache_mutex.tryLock());
    // A release/trim during contention must not spin or leak the released mapping.
    small.release();
    const released = trimCache();
    cache_mutex.unlock();
    try t.expectEqual(@as(usize, 0), released);
    try t.expectEqual(@as(usize, 0), trimCache());
}

// Host function returning a constant, used to detect stale instruction-cache contents when
// another thread returns a previously executable mapping to the shared cache.
fn cacheTestEmit(cb: *CodeBuffer, value: u16) !void {
    cb.reset();
    try cb.ensure(8);
    switch (builtin.cpu.arch) {
        .aarch64 => {
            cb.put32(0xd2800000 | (@as(u32, value) << 5)); // movz x0, value
            cb.put32(0xd65f03c0); // ret
        },
        .x86, .x86_64 => {
            cb.put8(0xb8); // mov eax, value
            cb.put32(value);
            cb.put8(0xc3);
        },
        .riscv64 => {
            cb.put32(0x00000513 | (@as(u32, value) << 20)); // addi a0, zero, value
            cb.put32(0x00008067); // ret
        },
        else => return error.UnsupportedHost,
    }
}

fn cacheStress(seed: u16, failed: *std.atomic.Value(bool)) void {
    cacheStressChecked(seed) catch failed.store(true, .release);
}

fn cacheStressChecked(seed: u16) !void {
    var cb = CodeBuffer.init(std.heap.page_allocator);
    defer cb.deinit();
    try cacheTestEmit(&cb, seed);
    const held = try add(&cb);
    defer held.release();
    const F = *const fn () callconv(.c) u32;
    for (0..200) |i| {
        const n: u16 = seed + @as(u16, @intCast(i));
        try cacheTestEmit(&cb, n);
        const f = try add(&cb);
        defer f.release();
        if (f.ptr(F)() != n or held.ptr(F)() != seed) return error.WrongResult;
        if (i % 7 == 0) _ = trimCache();
    }
}

test "runtime cache: concurrent reuse, live calls and trim preserve code" {
    if (builtin.cpu.arch != .aarch64 and builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .x86 and builtin.cpu.arch != .riscv64)
        return error.SkipZigTest;
    _ = trimCache();
    defer _ = trimCache();
    var failed = std.atomic.Value(bool).init(false);
    var threads: [4]std.Thread = undefined;
    var started: usize = 0;
    errdefer for (threads[0..started]) |thread| thread.join();
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, cacheStress, .{ @as(u16, @intCast(i * 256)), &failed });
        started += 1;
    }
    for (threads) |thread| thread.join();
    started = 0;
    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expect(trimCache() <= cache_capacity * page);
}
