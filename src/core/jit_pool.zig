//! Shared executable memory pool (AsmJit: JitAllocator, simplified).
//!
//! `JitRuntime.add` maps a fresh 16 KiB region per function (mmap + mprotect + munmap, ~3.5 µs on
//! Apple Silicon, and at least one page per function). `Pool` maps large chunks once and appends
//! functions into them (16-byte aligned), so installing a function is a memcpy plus an icache
//! flush. A chunk is unmapped when every function in it is released, or by `deinit`.
//!
//! W^X per host:
//! - macOS/arm64: MAP_JIT chunk, written inside a `pthread_jit_write_protect_np(0)` window of the
//!   calling thread only (other threads keep executing).
//! - other POSIX (Linux, macOS/x86-64 native): dual mapping (`JitRuntime.allocDual`), writes go
//!   through the RW view, the RX view is never writable.
//! - Rosetta 2 and Windows: one RX mapping, switched to RW for the copy and back to RX. While the
//!   copy runs the chunk is not executable, so do not `add` while another thread runs code from
//!   the same pool (use one pool per thread there).
//!
//! Not thread-safe: use one `Pool` per thread or lock around `add`/`release`.
const std = @import("std");
const builtin = @import("builtin");
const CodeBuffer = @import("code_buffer.zig").CodeBuffer;
const rt = @import("jit_runtime.zig");

extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn mprotect(addr: *anyopaque, len: usize, prot: c_int) c_int;
extern "c" fn pthread_jit_write_protect_np(enabled: c_int) void;
extern "c" fn sys_icache_invalidate(start: *anyopaque, len: usize) void;
extern "c" fn __clear_cache(start: *anyopaque, end: *anyopaque) void;
const PAGE_READWRITE = 0x04;
const PAGE_EXECUTE_READ = 0x20;
const MEM_COMMIT = 0x1000;
const MEM_RESERVE = 0x2000;
const MEM_RELEASE = 0x8000;
extern "kernel32" fn VirtualAlloc(addr: ?*anyopaque, size: usize, kind: u32, protect: u32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn VirtualFree(addr: *anyopaque, size: usize, kind: u32) callconv(.winapi) i32;
extern "kernel32" fn VirtualProtect(addr: *anyopaque, size: usize, protect: u32, old: *u32) callconv(.winapi) i32;
extern "kernel32" fn FlushInstructionCache(process: *anyopaque, addr: ?*const anyopaque, size: usize) callconv(.winapi) i32;
extern "kernel32" fn GetCurrentProcess() callconv(.winapi) *anyopaque;

const is_windows = builtin.os.tag == .windows;
const is_darwin = builtin.os.tag.isDarwin();
const use_map_jit = is_darwin and builtin.cpu.arch == .aarch64;
const PROT_READ = 1;
const PROT_WRITE = 2;
const PROT_EXEC = 4;
const MAP_PRIVATE = 0x02;
const MAP_ANON: c_int = if (is_darwin) 0x1000 else 0x20;
const MAP_JIT = 0x800;
const MAP_FAILED: usize = std.math.maxInt(usize);
const page = 16384;

pub const Error = rt.Error || error{OutOfMemory};

const Mode = enum { map_jit, dual, toggle };

const Chunk = struct {
    rx: []align(4096) u8,
    rw: []align(4096) u8, // == rx except in dual mode
    used: usize = 0,
    live: u32 = 0,
};

/// One installed function: `ptr` is executable, `len` its code size.
pub const Function = struct {
    addr: [*]const u8,
    len: usize,
    chunk: u32,
    pub fn ptr(self: Function, comptime F: type) F {
        return @ptrCast(@alignCast(@constCast(self.addr)));
    }
};

pub const Options = struct {
    /// Chunk size (rounded up to 16 KiB); larger functions get a chunk of their own.
    chunk_size: usize = 256 * 1024,
    /// Alignment of every function start.
    alignment: usize = 16,
};

pub const Pool = struct {
    gpa: std.mem.Allocator,
    opts: Options,
    mode: Mode,
    chunks: std.ArrayList(?Chunk) = .empty,
    /// Bytes mapped / bytes of code installed (stats).
    mapped: usize = 0,
    installed: usize = 0,

    pub fn init(gpa: std.mem.Allocator, opts: Options) Pool {
        const mode: Mode = if (use_map_jit) .map_jit else if (is_windows or rt.underRosetta() or !rt.dual_mapping_supported) .toggle else .dual;
        return .{ .gpa = gpa, .opts = opts, .mode = mode };
    }

    pub fn deinit(p: *Pool) void {
        for (p.chunks.items) |c| if (c) |ch| unmapChunk(p.mode, ch);
        p.chunks.deinit(p.gpa);
        p.* = undefined;
    }

    fn unmapChunk(mode: Mode, ch: Chunk) void {
        if (is_windows) {
            _ = VirtualFree(ch.rx.ptr, 0, MEM_RELEASE);
            return;
        }
        _ = munmap(ch.rx.ptr, ch.rx.len);
        if (mode == .dual) _ = munmap(ch.rw.ptr, ch.rw.len);
    }

    fn newChunk(p: *Pool, min: usize) Error!u32 {
        const size = std.mem.alignForward(usize, @max(min, p.opts.chunk_size), page);
        var ch: Chunk = undefined;
        switch (p.mode) {
            .dual => {
                const d = try rt.allocDual(size);
                ch = .{ .rx = d.rx, .rw = d.rw };
            },
            .map_jit, .toggle => {
                var mem: []align(4096) u8 = undefined;
                if (is_windows) {
                    const q = VirtualAlloc(null, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) orelse return error.MapFailed;
                    mem = @as([*]align(4096) u8, @ptrCast(@alignCast(q)))[0..size];
                    rt.fillTrap(mem);
                    var old: u32 = 0;
                    if (VirtualProtect(q, size, PAGE_EXECUTE_READ, &old) == 0) return error.ProtectFailed;
                } else {
                    const flags: c_int = MAP_PRIVATE | MAP_ANON | (if (p.mode == .map_jit) @as(c_int, MAP_JIT) else @as(c_int, 0));
                    const prot: c_int = if (p.mode == .map_jit) @as(c_int, PROT_READ | PROT_WRITE | PROT_EXEC) else @as(c_int, PROT_READ | PROT_WRITE);
                    const q = mmap(null, size, prot, flags, -1, 0) orelse return error.MapFailed;
                    if (@intFromPtr(q) == MAP_FAILED) return error.MapFailed;
                    mem = @as([*]align(4096) u8, @ptrCast(@alignCast(q)))[0..size];
                    if (p.mode == .map_jit) {
                        pthread_jit_write_protect_np(0);
                        rt.fillTrap(mem);
                        pthread_jit_write_protect_np(1);
                    } else {
                        rt.fillTrap(mem);
                        if (mprotect(mem.ptr, size, PROT_READ | PROT_EXEC) != 0) {
                            _ = munmap(mem.ptr, size);
                            return error.ProtectFailed;
                        }
                    }
                }
                ch = .{ .rx = mem, .rw = mem };
            },
        }
        p.mapped += size;
        // Reuse a released slot.
        for (p.chunks.items, 0..) |c, i| if (c == null) {
            p.chunks.items[i] = ch;
            return @intCast(i);
        };
        p.chunks.append(p.gpa, ch) catch |e| {
            unmapChunk(p.mode, ch);
            return e;
        };
        return @intCast(p.chunks.items.len - 1);
    }

    /// Copies finalized code into the pool and returns its executable address.
    pub fn add(p: *Pool, cb: *const CodeBuffer) Error!Function {
        return p.addBytes(cb.code());
    }

    pub fn addBytes(p: *Pool, code: []const u8) Error!Function {
        const n = @max(code.len, 1);
        // Last chunk with room (functions are appended; no hole reuse inside a chunk).
        var ci: ?u32 = null;
        var off: usize = 0;
        if (p.chunks.items.len != 0) {
            const last: u32 = @intCast(p.chunks.items.len - 1);
            if (p.chunks.items[last]) |ch| {
                const o = std.mem.alignForward(usize, ch.used, p.opts.alignment);
                if (o + n <= ch.rx.len) {
                    ci = last;
                    off = o;
                }
            }
        }
        if (ci == null) {
            ci = try p.newChunk(n);
            off = 0;
        }
        const ch = &p.chunks.items[ci.?].?;
        try write(p.mode, ch, off, code);
        ch.used = off + n;
        ch.live += 1;
        p.installed += code.len;
        return .{ .addr = ch.rx.ptr + off, .len = code.len, .chunk = ci.? };
    }

    fn write(mode: Mode, ch: *Chunk, off: usize, code: []const u8) Error!void {
        const dst_rx = ch.rx[off..][0..code.len];
        switch (mode) {
            .map_jit => {
                pthread_jit_write_protect_np(0);
                @memcpy(ch.rw[off..][0..code.len], code);
                pthread_jit_write_protect_np(1);
            },
            .dual => @memcpy(ch.rw[off..][0..code.len], code),
            .toggle => {
                // Pages covering the copy: RW, copy, RX.
                const lo = std.mem.alignBackward(usize, off, 4096);
                const hi = std.mem.alignForward(usize, off + code.len, 4096);
                const span = ch.rx[lo..hi];
                if (is_windows) {
                    var old: u32 = 0;
                    if (VirtualProtect(span.ptr, span.len, PAGE_READWRITE, &old) == 0) return error.ProtectFailed;
                    @memcpy(ch.rw[off..][0..code.len], code);
                    if (VirtualProtect(span.ptr, span.len, PAGE_EXECUTE_READ, &old) == 0) return error.ProtectFailed;
                    _ = FlushInstructionCache(GetCurrentProcess(), dst_rx.ptr, code.len);
                    return;
                }
                if (mprotect(span.ptr, span.len, PROT_READ | PROT_WRITE) != 0) return error.ProtectFailed;
                @memcpy(ch.rw[off..][0..code.len], code);
                if (mprotect(span.ptr, span.len, PROT_READ | PROT_EXEC) != 0) return error.ProtectFailed;
            },
        }
        if (code.len != 0) switch (builtin.cpu.arch) {
            .aarch64, .riscv64, .riscv32 => if (is_darwin) sys_icache_invalidate(@ptrCast(@constCast(dst_rx.ptr)), code.len) else __clear_cache(@ptrCast(dst_rx.ptr), @ptrCast(dst_rx.ptr + code.len)),
            else => {},
        };
    }

    /// Releases one function; its chunk is unmapped when nothing in it is live any more
    /// (the current, partly filled chunk is kept for further appends).
    pub fn release(p: *Pool, f: Function) void {
        const slot = &p.chunks.items[f.chunk];
        const ch = &slot.*.?;
        ch.live -= 1;
        if (ch.live == 0) {
            if (f.chunk + 1 == p.chunks.items.len) {
                // Rewinding re-uses addresses that already ran; under Rosetta keep appending.
                if (p.mode == .toggle) return;
                ch.used = 0; // rewind the tail chunk; stale code stays trap-free only up to `used`
                return;
            }
            p.mapped -= ch.rx.len;
            unmapChunk(p.mode, ch.*);
            slot.* = null;
        }
    }
};

// ---- tests ------------------------------------------------------------------------------
const t = std.testing;

fn retConst(arch: std.Target.Cpu.Arch, v: u16, out: *[8]u8) []const u8 {
    switch (arch) {
        .aarch64 => {
            std.mem.writeInt(u32, out[0..4], 0xD2800000 | @as(u32, v) << 5, .little); // movz x0, #v
            std.mem.writeInt(u32, out[4..8], 0xD65F03C0, .little); // ret
            return out[0..8];
        },
        .x86_64 => {
            out[0..6].* = .{ 0xB8, @truncate(v), @truncate(v >> 8), 0, 0, 0xC3 }; // mov eax, v; ret
            return out[0..6];
        },
        else => return out[0..0],
    }
}

test "jit pool: many functions share one chunk, all callable, release and reuse" {
    const arch = builtin.cpu.arch;
    if (arch != .aarch64 and arch != .x86_64) return error.SkipZigTest;
    var p = Pool.init(t.allocator, .{ .chunk_size = 16384 });
    defer p.deinit();
    var fs: [3000]Function = undefined;
    for (&fs, 0..) |*f, i| {
        var buf: [8]u8 = undefined;
        f.* = try p.addBytes(retConst(arch, @intCast(i), &buf));
    }
    try t.expectEqual(@as(usize, 3 * 16384), p.mapped); // 3000 x 16 B = 48000 B in three 16 KiB chunks
    for (fs, 0..) |f, i| try t.expectEqual(@as(u64, i), f.ptr(*const fn () callconv(.c) u64)() & 0xFFFF);
    for (fs) |f| p.release(f);
    var buf: [8]u8 = undefined;
    const g = try p.addBytes(retConst(arch, 77, &buf));
    try t.expectEqual(@as(u64, 77), g.ptr(*const fn () callconv(.c) u64)() & 0xFFFF);
    p.release(g);
}

test "jit pool: function larger than a chunk gets its own chunk" {
    const arch = builtin.cpu.arch;
    if (arch != .aarch64 and arch != .x86_64) return error.SkipZigTest;
    var p = Pool.init(t.allocator, .{ .chunk_size = 16384 });
    defer p.deinit();
    const big = try t.allocator.alloc(u8, 40000);
    defer t.allocator.free(big);
    var buf: [8]u8 = undefined;
    const tail = retConst(arch, 5, &buf);
    // nop sled then `ret 5`
    if (arch == .aarch64) {
        var i: usize = 0;
        while (i < big.len) : (i += 4) std.mem.writeInt(u32, big[i..][0..4], 0xD503201F, .little);
    } else @memset(big, 0x90);
    @memcpy(big[big.len - tail.len ..], tail);
    const f = try p.addBytes(big);
    try t.expectEqual(@as(u64, 5), f.ptr(*const fn () callconv(.c) u64)() & 0xFFFF);
}
