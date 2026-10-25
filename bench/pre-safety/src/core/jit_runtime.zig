//! Executable memory (AsmJit: JitRuntime + VirtMem, without the pooled JitAllocator).
//! macOS/arm64: MAP_JIT + pthread_jit_write_protect_np + sys_icache_invalidate.
//! Other POSIX hosts: mmap RW, copy, mprotect RX (W^X, never RWX).
const std = @import("std");
const builtin = @import("builtin");
const CodeBuffer = @import("code_buffer.zig").CodeBuffer;

extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn mprotect(addr: *anyopaque, len: usize, prot: c_int) c_int;
extern "c" fn pthread_jit_write_protect_np(enabled: c_int) void;
extern "c" fn sys_icache_invalidate(start: *anyopaque, len: usize) void;
extern "c" fn __clear_cache(start: *anyopaque, end: *anyopaque) void;

const PROT_READ = 1;
const PROT_WRITE = 2;
const PROT_EXEC = 4;
const is_darwin = builtin.os.tag.isDarwin();
const MAP_PRIVATE = 0x02;
const MAP_ANON: c_int = if (is_darwin) 0x1000 else 0x20;
const MAP_JIT = 0x800;
const use_map_jit = is_darwin and builtin.cpu.arch == .aarch64;
const MAP_FAILED: usize = std.math.maxInt(usize);

pub const Error = error{ MapFailed, ProtectFailed };

pub const Function = struct {
    mem: []align(4096) u8,
    pub fn ptr(self: Function, comptime F: type) F {
        return @ptrCast(@alignCast(self.mem.ptr));
    }
    pub fn release(self: Function) void {
        _ = munmap(self.mem.ptr, self.mem.len);
    }
};

/// Allocation granule: 16 KiB covers both 4K (x86) and 16K (Apple arm64) pages.
const page = 16384;

/// Copies finalized code into fresh executable memory.
pub fn add(cb: *const CodeBuffer) Error!Function {
    const code = cb.code();
    const size = std.mem.alignForward(usize, @max(code.len, 1), page);
    const flags: c_int = MAP_PRIVATE | MAP_ANON | (if (use_map_jit) MAP_JIT else 0);
    const prot: c_int = if (use_map_jit) PROT_READ | PROT_WRITE | PROT_EXEC else PROT_READ | PROT_WRITE;
    const p = mmap(null, size, prot, flags, -1, 0) orelse return error.MapFailed;
    if (@intFromPtr(p) == MAP_FAILED) return error.MapFailed;
    const mem: []align(4096) u8 = @as([*]align(4096) u8, @ptrCast(@alignCast(p)))[0..size];
    if (use_map_jit) {
        pthread_jit_write_protect_np(0); // this thread: writable
        @memcpy(mem[0..code.len], code);
        pthread_jit_write_protect_np(1); // this thread: executable
        sys_icache_invalidate(mem.ptr, code.len);
    } else {
        @memcpy(mem[0..code.len], code);
        if (mprotect(mem.ptr, size, PROT_READ | PROT_EXEC) != 0) {
            _ = munmap(mem.ptr, size);
            return error.ProtectFailed;
        }
        if (builtin.cpu.arch == .aarch64) __clear_cache(mem.ptr, mem.ptr + code.len);
    }
    return .{ .mem = mem };
}
