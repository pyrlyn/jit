//! Executable memory (AsmJit: JitRuntime + VirtMem, without the pooled JitAllocator).
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
const use_map_jit = is_darwin and builtin.cpu.arch == .aarch64;
const MAP_FAILED: usize = std.math.maxInt(usize);

pub const Error = error{ MapFailed, ProtectFailed, OutOfBounds };

pub const Function = struct {
    mem: []align(4096) u8,
    pub fn ptr(self: Function, comptime F: type) F {
        return @ptrCast(@alignCast(self.mem.ptr));
    }
    pub fn release(self: Function) void {
        unmap(self.mem.ptr, self.mem.len);
    }
};

/// Allocation granule: 16 KiB covers both 4K (x86) and 16K (Apple arm64) pages.
const page = 16384;

/// Copies finalized code into fresh executable memory.
pub fn add(cb: *const CodeBuffer) Error!Function {
    const code = cb.code();
    const size = std.mem.alignForward(usize, @max(code.len, 1), page);
    if (is_windows) return addWindows(code, size);
    const flags: c_int = MAP_PRIVATE | MAP_ANON | (if (use_map_jit) MAP_JIT else 0);
    const prot: c_int = if (use_map_jit) PROT_READ | PROT_WRITE | PROT_EXEC else PROT_READ | PROT_WRITE;
    const p = mmap(null, size, prot, flags, -1, 0) orelse return error.MapFailed;
    if (@intFromPtr(p) == MAP_FAILED) return error.MapFailed;
    const mem: []align(4096) u8 = @as([*]align(4096) u8, @ptrCast(@alignCast(p)))[0..size];
    if (use_map_jit) {
        // MAP_JIT pages are RWX in the page tables, but Apple Silicon enforces W^X per thread:
        // they are writable only inside this window and only for this thread.
        pthread_jit_write_protect_np(0); // this thread: writable, not executable
        defer pthread_jit_write_protect_np(1); // this thread: executable, not writable
        @memcpy(mem[0..code.len], code);
        fillTrap(mem[code.len..]);
    }
    if (use_map_jit) {
        sys_icache_invalidate(mem.ptr, code.len);
    } else {
        @memcpy(mem[0..code.len], code);
        fillTrap(mem[code.len..]);
        if (mprotect(mem.ptr, size, PROT_READ | PROT_EXEC) != 0) {
            _ = munmap(mem.ptr, size);
            return error.ProtectFailed;
        }
        if (builtin.cpu.arch == .aarch64) __clear_cache(mem.ptr, mem.ptr + code.len);
    }
    return .{ .mem = mem };
}

fn addWindows(code: []const u8, size: usize) Error!Function {
    const p = VirtualAlloc(null, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) orelse return error.MapFailed;
    const mem: []align(4096) u8 = @as([*]align(4096) u8, @ptrCast(@alignCast(p)))[0..size];
    @memcpy(mem[0..code.len], code);
    fillTrap(mem[code.len..]);
    var old: u32 = 0;
    if (VirtualProtect(p, size, PAGE_EXECUTE_READ, &old) == 0) {
        _ = VirtualFree(p, 0, MEM_RELEASE);
        return error.ProtectFailed;
    }
    _ = FlushInstructionCache(GetCurrentProcess(), p, code.len);
    return .{ .mem = mem };
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
    if (builtin.cpu.arch != .aarch64) return;
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
