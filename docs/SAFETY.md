# Safety


Validation and emission hooks. Both are optional, chosen at comptime, and free when off.

`Assembler` is `AssemblerT(default_options)`. Build your own variant with:

```zig
const A = pjit.x86_64.AssemblerT(.{
    .validate = true,              // check operands before encoding
    .Hooks = pjit.x86_64.CountingHooks, // or void
});
var counter: pjit.x86_64.CountingHooks = .{};
const a = A.initHooks(&cb, &counter);   // A.init(&cb) = hooks type set but no instance
```

## Validation
With `.validate = true`, every emitter checks its operands and returns an error instead of
emitting wrong bytes. Nothing is written when a check fails.

| Backend | Checks | Error |
|---|---|---|
| x86-64 | `rsp` as index (unencodable: SIB index 100 means "none") | `InvalidOperand` |
| x86-64 | scale other than `.x1` without an index | `InvalidOperand` |
| both | label not created by this buffer (`jmp`, `call`, `jcc`, `b`, `bind`, ...) | `InvalidLabel` |
| AArch64 | register 31 (`.zr_sp`) where it would silently mean XZR: register-form `add/sub/mul/cmp`, `movz/movk`, `mov` destination, `ldr/str` data register, `blr`, `cbz/cbnz` | `InvalidOperand` |
| AArch64 | backward branch out of range (`b.cond`/`cbz`: ±1 MiB, `b`/`bl`: ±128 MiB) | `DisplacementOutOfRange` |
| AArch64 | misaligned `ldr/str` offset (checked always) | `InvalidImmediate` |

Immediate widths are enforced by the types (`i32` for imm32, `u12`, `u16`, `u15`, comptime
`shift`), so out-of-range literals fail to compile and runtime values must be cast.

```zig
const a = pjit.x86_64.AssemblerT(.{ .validate = true }).init(&cb);
a.movRM(.rax, pjit.x86_64.ptrIdx(.rdi, .rsp, .x1, 0)) catch |e| {
    // e == error.InvalidOperand
};
```

## Hooks
A hook type is a struct with optional methods. They are called directly (inlined), not
through function pointers.

```zig
pub fn pre(self: *H, ev: PreEvent) Error!void // before encoding; return an error to veto
pub fn post(self: *H, ev: Event) void         // after encoding
```

`Event` has `id: Inst` (e.g. `.mov_rr`, `.alu_ri`, `.add_imm`), `ops: Ops` (registers, memory,
immediate, label, ALU op / condition; unused fields are null), `offset: u32`, and
`bytes: []const u8` (the encoded instruction). `PreEvent` is the same without `bytes`.

Ready-made hooks (in both backends and in `pjit.safety`):

| Hook | Purpose |
|---|---|
| `CountingHooks` | counts instructions and bytes |
| `LogHooks(cap)` | text log `0004  mov_rr   48 89 c8` into a fixed buffer (`.text()`) |
| `CallbackHooks` | runtime function pointer + context, e.g. for C callers (one indirect call per instruction) |

Custom hook example (logging + per-instruction counters), see `examples/example_hooks.zig`:

```zig
const Stats = struct {
    per_id: std.EnumArray(x.Inst, u32) = .initFill(0),
    pub fn post(self: *Stats, ev: x.Event) void {
        self.per_id.getPtr(ev.id).* += 1;
    }
};
var stats: Stats = .{};
const a = x.AssemblerT(.{ .Hooks = Stats }).initHooks(&cb, &stats);
```

Composite helpers report their parts: `callAbs` = `mov_ri` + `call_r`, `movImm` = `movz` + `movk`s.

## Enable / disable

| Build | `validate` default | `hooks` default |
|---|---|---|
| `Debug`, `ReleaseSafe` | on | on |
| `ReleaseFast`, `ReleaseSmall` | off | **on** |

Override with `-Dvalidate=true|false` and `-Dhooks=true|false`. They set
`default_options` of the default `Assembler` (`hooks` means "supports a `CallbackHooks`
instance"). `AssemblerT(.{...})` ignores the build flags, so code that needs a fixed
configuration can choose it explicitly.

Hooks stay compiled in for release builds so a hook can always be attached; `-Dhooks=false`
is the explicit opt-out. With hooks compiled in but no instance attached, each instruction
does one null check. That is free when the compiler can see that nothing is attached
(e.g. `init(&cb)` in the same function), and costs up to ~0.6–1 ns/instr in label- and
memory-heavy code when it cannot (see bench/RESULTS.md). Use `-Dhooks=false` or
`AssemblerT(.{})` for the fastest path.

When a layer is off it disappears at compile time: no checks, no branches, no extra field
(`@sizeOf(AssemblerT(.{})) == @sizeOf(*CodeBuffer)`), identical bytes.

