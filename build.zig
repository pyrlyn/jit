const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const targets = b.option([]const u8, "targets", "Comma-separated backends to compile in (x86_64,aarch64)") orelse "x86_64,aarch64";

    const opts = b.addOptions();
    const want_x86_64 = std.mem.indexOf(u8, targets, "x86_64") != null;
    const want_aarch64 = std.mem.indexOf(u8, targets, "aarch64") != null;
    opts.addOption(bool, "x86_64", want_x86_64);
    opts.addOption(bool, "aarch64", want_aarch64);
    // Validation: on by default in Debug/ReleaseSafe, off in ReleaseFast/ReleaseSmall.
    const safe_default = optimize == .Debug or optimize == .ReleaseSafe;
    opts.addOption(bool, "validate", b.option(bool, "validate", "Validate operands before emission (default: on in Debug/ReleaseSafe)") orelse safe_default);
    // Hooks stay available in every mode (Ivan: not disabled in release); -Dhooks=false opts out.
    opts.addOption(bool, "hooks", b.option(bool, "hooks", "Default Assembler supports runtime emission hooks (default: on in all modes)") orelse true);
    // Benchmark only: attach a comptime CountingHooks instead of the default CallbackHooks.
    opts.addOption(bool, "asmjit_tests", b.option(bool, "asmjit-tests", "test: include AsmJit's ported assembler tests (src/asmjit_x64_tests.zig, src/asmjit_x86_tests.zig, src/asmjit_a64_tests.zig)") orelse false);
    opts.addOption(bool, "gp_fuzz", b.option(bool, "gp-fuzz", "test: include src/gp_fuzz.zig (written by scripts/fuzz_gp.py)") orelse false);
    opts.addOption(bool, "bench_counting", b.option(bool, "bench-counting", "bench: use comptime CountingHooks (implies validate setting as given)") orelse false);

    const pyrlyn_jit = b.addModule("pyrlyn_jit", .{
        .root_source_file = b.path("src/pyrlyn_jit.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    pyrlyn_jit.addOptions("pyrlyn_jit_options", opts);

    const tests = b.addTest(.{ .root_module = pyrlyn_jit });
    // Own fuzz driver: Zig 0.16.0's test runner does not build with `--fuzz` (see SECURITY.md).
    const fuzz_exe = b.addExecutable(.{ .name = "pyrlyn_jit_fuzz", .root_module = b.createModule(.{
        .root_source_file = b.path("src/fuzz_main.zig"),
        .target = target,
        .optimize = if (optimize == .Debug) .ReleaseSafe else optimize,
        .link_libc = true,
    }) });
    b.step("fuzz", "Fuzz the encoders/code buffer with random seeds (env: PYRLYN_JIT_FUZZ_ROUNDS/SEED/ITERS)").dependOn(&b.addRunArtifact(fuzz_exe).step);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // ---- C ABI library (include/pyrlyn_jit.h) + examples in C, C++, Rust, Zig ----------------
    const capi_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    capi_mod.addOptions("pyrlyn_jit_options", opts);
    const lib_static = b.addLibrary(.{ .name = "pyrlyn-jit", .linkage = .static, .root_module = capi_mod });
    const lib_shared = b.addLibrary(.{ .name = "pyrlyn-jit", .linkage = .dynamic, .root_module = capi_mod });
    b.installArtifact(lib_shared);
    b.installDirectory(.{ .source_dir = b.path("include"), .install_dir = .header, .install_subdir = "" });
    // pkg-config file (zig-out/lib/pkgconfig/pyrlyn-jit.pc); prefix is relocatable via pcfiledir.
    const pc_version = b.option([]const u8, "version", "Version written to pyrlyn-jit.pc") orelse "0.1.0";
    const pc = b.addWriteFiles().add("pyrlyn-jit.pc", b.fmt(
        \\prefix=${{pcfiledir}}/../..
        \\libdir=${{prefix}}/lib
        \\includedir=${{prefix}}/include
        \\
        \\Name: pyrlyn-jit
        \\Description: pyrlyn-jit machine-code generator (C ABI)
        \\Version: {s}
        \\Libs: -L${{libdir}} -lpyrlyn-jit
        \\Cflags: -I${{includedir}}
        \\
    , .{pc_version}));
    b.getInstallStep().dependOn(&b.addInstallFileWithDir(pc, .lib, "pkgconfig/pyrlyn-jit.pc").step);
    b.step("lib", "Build libpyrlyn-jit (static + shared) and install include/").dependOn(&b.addInstallArtifact(lib_static, .{}).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = capi_mod })).step);
    // Plain object for foreign toolchains: Apple ld rejects Zig's archive members that are not
    // 8-byte aligned ("64-bit mach-o not 8-byte aligned"), so external linkers get the .o.
    // Separate module without stack probes: x86_64 Debug code otherwise references Zig's
    // __zig_probe_stack (compiler_rt), which foreign linkers (c++, rustc) cannot resolve.
    const obj_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .stack_check = false,
    });
    obj_mod.addOptions("pyrlyn_jit_options", opts);
    const lib_obj = b.addObject(.{ .name = "pyrlyn_jit", .root_module = obj_mod });

    // Installed libpyrlyn-jit.a: on macOS the archive is re-packed with the system `ar` from the plain
    // object, because Apple ld ignores Zig's unaligned archive members (see lib_obj above).
    // Zig's own consumers (examples below) still link lib_static directly.
    if (target.result.os.tag.isDarwin() and @import("builtin").os.tag.isDarwin()) {
        const ar = b.addSystemCommand(&.{ "ar", "-rcs" });
        const repacked = ar.addOutputFileArg("libpyrlyn-jit.a");
        ar.addFileArg(lib_obj.getEmittedBin());
        b.getInstallStep().dependOn(&b.addInstallFileWithDir(repacked, .lib, "libpyrlyn-jit.a").step);
    } else {
        b.installArtifact(lib_static);
    }

    const examples_step = b.step("examples", "Build and run examples/{c,cpp,zig} (+ rust with -Drust=true)");
    {
        const m = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        m.addCSourceFile(.{ .file = b.path("examples/c/main.c"), .flags = &.{"-std=c11"} });
        m.addIncludePath(b.path("include"));
        m.linkLibrary(lib_static);
        examples_step.dependOn(&b.addRunArtifact(b.addExecutable(.{ .name = "example_c", .root_module = m })).step);
    }
    // C++ is built with the system compiler (`c++`, override with -Dcxx=...): Zig 0.16's bundled
    // libc++ does not compile against current macOS SDKs.
    {
        const cxx = b.addSystemCommand(&.{ b.option([]const u8, "cxx", "examples: C++ compiler (default: c++)") orelse "c++", "-std=c++17", "-O1", "-Wl,-w" }); // -w: Zig stamps the .o with the patch-level macOS version
        // Cross-arch on macOS (e.g. -Dtarget=x86_64-macos under Rosetta): link the example for the target arch.
        if (target.result.os.tag == .macos and target.result.cpu.arch != @import("builtin").cpu.arch) {
            cxx.addArgs(&.{ "-arch", switch (target.result.cpu.arch) {
                .x86_64 => "x86_64",
                .aarch64 => "arm64",
                else => @panic("examples: unsupported macOS arch for c++"),
            } });
        }
        cxx.addPrefixedDirectoryArg("-I", b.path("include"));
        cxx.addFileArg(b.path("examples/cpp/main.cpp"));
        cxx.addFileArg(lib_obj.getEmittedBin());
        cxx.addArg("-o");
        const cpp_exe = cxx.addOutputFileArg("example_cpp");
        const run_cpp = std.Build.Step.Run.create(b, "run example_cpp");
        run_cpp.addFileArg(cpp_exe);
        examples_step.dependOn(&run_cpp.step);
    }
    const zig_example = b.addExecutable(.{ .name = "example_zig", .root_module = b.createModule(.{
        .root_source_file = b.path("examples/zig/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "pyrlyn_jit", .module = pyrlyn_jit }},
    }) });
    examples_step.dependOn(&b.addRunArtifact(zig_example).step);
    if (b.option(bool, "rust", "examples: also build and run examples/rust with rustc (no crates)") orelse false) {
        const rustc = b.addSystemCommand(&.{ "rustc", "--edition=2021", "-O", "-o" });
        const rust_exe = rustc.addOutputFileArg("example_rust");
        rustc.addFileArg(b.path("examples/rust/main.rs"));
        rustc.addPrefixedFileArg("-Clink-arg=", lib_obj.getEmittedBin());
        const run_rust = std.Build.Step.Run.create(b, "run example_rust");
        run_rust.addFileArg(rust_exe);
        examples_step.dependOn(&run_rust.step);
    }
    test_step.dependOn(examples_step);

    inline for (.{ "example_x86_64", "example_aarch64", "example_hooks", "bench_x86_64" }) |name| {
        const enabled = if (comptime std.mem.eql(u8, name, "example_hooks")) want_x86_64 and want_aarch64 else if (comptime std.mem.endsWith(u8, name, "aarch64")) want_aarch64 else want_x86_64;
        const dir = if (comptime std.mem.startsWith(u8, name, "bench")) "bench/" else "examples/";
        if (enabled) {
            const exe = b.addExecutable(.{
                .name = name,
                .root_module = b.createModule(.{
                    .root_source_file = b.path(dir ++ name ++ ".zig"),
                    .target = target,
                    .optimize = optimize,
                    .link_libc = true,
                    .imports = &.{.{ .name = "pyrlyn_jit", .module = pyrlyn_jit }},
                }),
            });
            b.installArtifact(exe);
            if (comptime std.mem.startsWith(u8, name, "bench")) {
                // Harness tests: scenarios must stay byte-identical to AsmJit's output.
                const bt = b.addTest(.{ .root_module = exe.root_module });
                test_step.dependOn(&b.addRunArtifact(bt).step);
            }
            const run = b.addRunArtifact(exe);
            if (b.args) |args| run.addArgs(args);
            b.step("run-" ++ name, "Run " ++ name).dependOn(&run.step);
        }
    }
}
