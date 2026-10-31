//! Standalone fuzz driver (`zig build fuzz`). Runs the harnesses from src/fuzz.zig over fresh
//! random seeds in a loop; an alternative to `zig build test --fuzz` (works since Zig 0.17.0)
//! that needs no fuzzer UI and replays by seed. Safety checks stay on (Debug/ReleaseSafe); a crash prints the seed, and
//! `PYRLYN_JIT_FUZZ_SEED=<seed> zig build fuzz` replays it.
//!
//! Environment: PYRLYN_JIT_FUZZ_ROUNDS (default 20; 0 = run forever), PYRLYN_JIT_FUZZ_SEED (start seed),
//! PYRLYN_JIT_FUZZ_ITERS (cases per harness per round, default 100000).
const std = @import("std");
const fuzz = @import("fuzz.zig");
extern "c" fn time(t: ?*i64) i64;
const c_time = time;

fn envInt(name: [*:0]const u8, default: u64) u64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 0) catch default;
}

pub fn main() !void {
    const rounds = envInt("PYRLYN_JIT_FUZZ_ROUNDS", 20);
    const iters = envInt("PYRLYN_JIT_FUZZ_ITERS", 100_000);
    var seed = envInt("PYRLYN_JIT_FUZZ_SEED", @as(u64, @intCast(c_time(null))) *% 0x9E3779B97F4A7C15);
    var round: u64 = 0;
    while (rounds == 0 or round < rounds) : (round += 1) {
        inline for (.{ .{ "a64", fuzz.a64One }, .{ "x86-64 gp", fuzz.gpOne }, .{ "code buffer", fuzz.bufferOne } }) |h| {
            std.debug.print("round {d} {s}: seed=0x{x}\n", .{ round, h[0], seed });
            fuzz.runPrng(h[1], iters, seed) catch |e| {
                std.debug.print("FAILURE in {s} harness, replay with PYRLYN_JIT_FUZZ_SEED=0x{x} PYRLYN_JIT_FUZZ_ROUNDS=1: {s}\n", .{ h[0], seed, @errorName(e) });
                return e;
            };
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
        }
    }
    std.debug.print("fuzz: {d} rounds x 3 harnesses x {d} cases, no failures\n", .{ round, iters });
}
