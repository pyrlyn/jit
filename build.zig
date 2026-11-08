const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const targets = b.option([]const u8, "targets", "Comma-separated backends to compile in (x86_64,aarch64)") orelse "x86_64,aarch64";

    const opts = b.addOptions();
    const llvm_prefix = b.option([]const u8, "llvm-prefix", "Enable the optional LLVM backend using PREFIX/include and PREFIX/lib (shared libLLVM)");
    opts.addOption(bool, "llvm", llvm_prefix != null);
    const want_bc2jit = b.option(bool, "bc2jit", "Include the KXBC Zig/C API and CLI (default: true)") orelse true;
    opts.addOption(bool, "bc2jit", want_bc2jit);
    const want_x86_64 = std.mem.indexOf(u8, targets, "x86_64") != null;
    const want_aarch64 = std.mem.indexOf(u8, targets, "aarch64") != null;
    opts.addOption(bool, "x86_64", want_x86_64);
    opts.addOption(bool, "aarch64", want_aarch64);
    // Validation: on by default in Debug/ReleaseSafe, off in ReleaseFast/ReleaseSmall.
    const safe_default = optimize == .debug or optimize == .safe;
    opts.addOption(bool, "validate", b.option(bool, "validate", "Validate operands before emission (default: on in Debug/ReleaseSafe)") orelse safe_default);
    // Hooks stay available in every mode (Ivan: not disabled in release); -Dhooks=false opts out.
    opts.addOption(bool, "hooks", b.option(bool, "hooks", "Default Assembler supports runtime emission hooks (default: on in all modes)") orelse true);
    // Benchmark only: attach a comptime CountingHooks instead of the default CallbackHooks.
    opts.addOption(bool, "asmjit_tests", b.option(bool, "asmjit-tests", "test: include AsmJit's ported assembler tests (src/asmjit_x64_tests.zig, src/asmjit_x86_tests.zig, src/asmjit_a64_tests.zig)") orelse false);
    // x86-64: restrict instruction forms to these CPU features at compile time (empty: all).
    opts.addOption([]const u8, "x86_features", b.option([]const u8, "x86-features", "x86-64: comma-separated CPU features (isa.Feature, e.g. sse4_2,avx2,avx512_f); forms needing others are removed and calling them is a compile error") orelse "");
    opts.addOption(bool, "gp_fuzz", b.option(bool, "gp-fuzz", "test: include src/gp_fuzz.zig (written by scripts/fuzz_gp.py)") orelse false);
    opts.addOption(bool, "bench_counting", b.option(bool, "bench-counting", "bench: use comptime CountingHooks (implies validate setting as given)") orelse false);

    const pyrlyn_jit = b.addModule("pyrlyn_jit", .{
        .root_source_file = b.path("src/pyrlyn_jit.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    pyrlyn_jit.addOptions("pyrlyn_jit_options", opts);
    if (llvm_prefix) |prefix| {
        const bindings = b.addTranslateC(.{
            .root_source_file = b.path("src/core/llvm_c.h"),
            .target = target,
            .optimize = optimize,
        });
        bindings.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        pyrlyn_jit.addImport("llvm_c", bindings.createModule());
        const lib: std.Build.LazyPath = .{ .cwd_relative = b.pathJoin(&.{ prefix, "lib" }) };
        pyrlyn_jit.addLibraryPath(lib);
        pyrlyn_jit.addRPath(lib);
        pyrlyn_jit.linkSystemLibrary("LLVM", .{ .use_pkg_config = .no });
    }

    const bc2jit = b.addModule("bc2jit", .{
        .root_source_file = b.path("bc2jit/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "pyrlyn_jit", .module = pyrlyn_jit }},
    });

    const tests = b.addTest(.{ .root_module = pyrlyn_jit });
    // Standalone fuzz driver (seeded, replayable); `zig build test --fuzz` also works (see SECURITY.md).
    const fuzz_exe = b.addExecutable(.{ .name = "pyrlyn_jit_fuzz", .root_module = b.createModule(.{
        .root_source_file = b.path("src/fuzz_main.zig"),
        .target = target,
        .optimize = if (optimize == .debug) .safe else optimize,
        .link_libc = true,
    }) });
    b.step("fuzz", "Fuzz the encoders/code buffer with random seeds (env: PYRLYN_JIT_FUZZ_ROUNDS/SEED/ITERS)").dependOn(&b.addRunArtifact(fuzz_exe).step);

    // ---- encoder tables from LLVM (tools/gen_tables.zig; docs/llvm-tables.md) -------------------
    // Not part of a normal build: the generated tables are committed. Needs an llvm-project
    // checkout (llvmorg-23.1.4) and llvm-tblgen.
    {
        const llvm_src = b.option([]const u8, "llvm-src", "gen-tables/check-tables: llvm-project checkout (llvmorg-23.1.4)");
        const tblgen = b.option([]const u8, "llvm-tblgen", "gen-tables/check-tables: llvm-tblgen binary (default: llvm-tblgen on PATH)") orelse "llvm-tblgen";
        const host_mod = struct {
            fn f(bb: *std.Build, path: []const u8) *std.Build.Module {
                return bb.createModule(.{ .root_source_file = bb.path(path), .target = bb.graph.host, .optimize = .ReleaseSafe });
            }
        }.f;
        const gen_exe = b.addExecutable(.{ .name = "gen_tables", .root_module = host_mod(b, "tools/gen_tables.zig") });
        const cmp_exe = b.addExecutable(.{ .name = "compare_tables", .root_module = host_mod(b, "tools/compare_tables.zig") });
        const gen_step = b.step("gen-tables", "Regenerate src/x86_64/{gp,vec}_table.zig and src/aarch64/a64_table.zig from LLVM (-Dllvm-src=...)");
        const check_step = b.step("check-tables", "Check that the committed LLVM-generated tables match a fresh generation (-Dllvm-src=...)");
        if (llvm_src) |src| {
            // target, LLVM target dir, top-level .td, generated table
            const llvm_targets = [_][4][]const u8{
                .{ "x86", "X86", "X86.td", "src/x86_64/gp_table.zig" },
                .{ "a64", "AArch64", "AArch64.td", "src/aarch64/a64_table.zig" },
            };
            for (llvm_targets) |t| {
                const tg = b.addSystemCommand(&.{ tblgen, "--dump-json" });
                tg.addArgs(&.{ "-I", b.pathJoin(&.{ src, "llvm/include" }), "-I", b.pathJoin(&.{ src, "llvm/lib/Target", t[1] }) });
                tg.addArg(b.pathJoin(&.{ src, "llvm/lib/Target", t[1], t[2] }));
                tg.addArg("-o");
                const json = tg.addOutputFileArg(b.fmt("{s}.json", .{t[1]}));

                const gen = b.addRunArtifact(gen_exe);
                gen.addArg(t[0]);
                gen.addFileArg(json);
                gen.addDirectoryArg(b.path("."));
                gen.has_side_effects = true;
                gen_step.dependOn(&gen.step);

                const tmp = b.addRunArtifact(gen_exe);
                tmp.addArg(t[0]);
                tmp.addFileArg(json);
                const out_dir = tmp.addOutputDirectoryArg("gen");
                const cmp = b.addRunArtifact(cmp_exe);
                cmp.addArg("--exact");
                cmp.addFileArg(b.path(t[3]));
                cmp.addFileArg(out_dir.path(b, t[3]));
                check_step.dependOn(&cmp.step);
                if (std.mem.eql(u8, t[0], "x86")) {
                    const cmp_vec = b.addRunArtifact(cmp_exe);
                    cmp_vec.addArg("--exact");
                    cmp_vec.addFileArg(b.path("src/x86_64/vec_table.zig"));
                    cmp_vec.addFileArg(out_dir.path(b, "src/x86_64/vec_table.zig"));
                    check_step.dependOn(&cmp_vec.step);
                }
            }
        } else {
            const fail = b.addFail("gen-tables/check-tables need -Dllvm-src=PATH to an llvm-project checkout (llvmorg-23.1.4)");
            gen_step.dependOn(&fail.step);
            check_step.dependOn(&fail.step);
        }
    }

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    if (want_bc2jit) @import("bc2jit/build_steps.zig").add(b, target, optimize, pyrlyn_jit, bc2jit, test_step, llvm_prefix != null);

    const runtime_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/core/jit_runtime.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) }));
    runtime_tests.skip_foreign_checks = true;
    b.step("test-jit-runtime", "Test executable mapping reuse, ownership and concurrent calls").dependOn(&runtime_tests.step);

    // 32-bit x86 Compiler execution tests (x86-linux; runs natively on x86_64 Linux kernels).
    const x86_32_mod = b.createModule(.{
        .root_source_file = b.path("src/x86_32_tests.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .x86, .os_tag = .linux }),
        .link_libc = true,
        .optimize = optimize,
    });
    x86_32_mod.addOptions("pyrlyn_jit_options", opts);
    b.step("test-x86-32", "Run the x86 (32-bit) Compiler execution tests as an x86-linux process").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = x86_32_mod })).step);

    // RISC-V 64 execution tests (riscv64-linux-musl, static so qemu-user needs no sysroot; run with `-fqemu` and qemu-riscv64 on PATH).
    const rv64_mod = b.createModule(.{
        .root_source_file = b.path("src/riscv_exec_tests.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .riscv64, .os_tag = .linux, .abi = .musl }),
        .link_libc = true,
        .optimize = optimize,
    });
    b.step("test-riscv64", "Run the RISC-V 64 execution tests as a riscv64-linux process (use -fqemu)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = rv64_mod })).step);

    // ---- C ABI library (include/pyrlyn_jit.h) + examples in C, C++, Rust, Zig ----------------
    const capi_mod = b.createModule(.{
        .root_source_file = b.path("src/library.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    capi_mod.addImport("pyrlyn_jit", pyrlyn_jit);
    capi_mod.addImport("bc2jit", bc2jit);
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
        .root_source_file = b.path("src/library.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .stack_check = false,
    });
    const obj_jit = b.createModule(.{
        .root_source_file = b.path("src/pyrlyn_jit.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .stack_check = false,
    });
    obj_jit.addOptions("pyrlyn_jit_options", opts);
    const obj_bc2jit = b.createModule(.{
        .root_source_file = b.path("bc2jit/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .stack_check = false,
        .imports = &.{.{ .name = "pyrlyn_jit", .module = obj_jit }},
    });
    obj_mod.addImport("pyrlyn_jit", obj_jit);
    obj_mod.addImport("bc2jit", obj_bc2jit);
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

    const bc_api_tests = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("bc2jit/api_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "bc2jit", .module = bc2jit }},
    }) }));
    bc_api_tests.skip_foreign_checks = true;
    test_step.dependOn(&bc_api_tests.step);
    b.step("test-bc2jit-api", "Test the public Zig bytecode API (also when disabled)").dependOn(&bc_api_tests.step);

    const bc_c_test = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    bc_c_test.addCSourceFile(.{ .file = b.path("bc2jit/test_api.c"), .flags = &.{"-std=c11"} });
    bc_c_test.addIncludePath(b.path("include"));
    bc_c_test.linkLibrary(lib_static);
    const run_bc_c = b.addRunArtifact(b.addExecutable(.{ .name = "test_bc2jit_c", .root_module = bc_c_test }));
    run_bc_c.skip_foreign_checks = true;
    test_step.dependOn(&run_bc_c.step);
    b.step("test-bc2jit-c", "Test the C bytecode API (also when disabled)").dependOn(&run_bc_c.step);

    const bc_bench = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    bc_bench.addCSourceFile(.{ .file = b.path("bench/bc2jit_api.c"), .flags = &.{"-std=c11"} });
    bc_bench.addIncludePath(b.path("include"));
    bc_bench.linkLibrary(lib_static);
    const bc_bench_exe = b.addExecutable(.{ .name = "bench_bc2jit_api", .root_module = bc_bench });
    b.step("bench-bc2jit-api", "Build the C API JIT benchmark (use -Doptimize=ReleaseFast)").dependOn(&b.addInstallArtifact(bc_bench_exe, .{}).step);

    const examples_step = b.step("examples", "Build and run examples/{c,cpp,zig} (+ rust with -Drust=true)");
    {
        const m = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        m.addCSourceFile(.{ .file = b.path("examples/c/main.c"), .flags = &.{"-std=c11"} });
        m.addIncludePath(b.path("include"));
        m.linkLibrary(lib_static);
        examples_step.dependOn(&b.addRunArtifact(b.addExecutable(.{ .name = "example_c", .root_module = m })).step);
    }
    // C++ is built with the system compiler (`c++`, override with -Dcxx=...). Zig 0.17's bundled
    // `zig c++` also works (via a wrapper script, e.g. -Dcxx=./zigcxx); the default is unchanged.
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
            run.addPassthruArgs();
            b.step("run-" ++ name, "Run " ++ name).dependOn(&run.step);
        }
    }

    // Benchmark regression tests (bench/bench_test.zig, baselines in bench/baseline-<arch>-<os>.txt).
    // Always ReleaseFast (library included) so numbers don't depend on -Doptimize.
    {
        const jit_fast = b.createModule(.{
            .root_source_file = b.path("src/pyrlyn_jit.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        jit_fast.addOptions("pyrlyn_jit_options", opts);
        const bt = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path("bench/bench_test.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "pyrlyn_jit", .module = jit_fast },
                .{ .name = "bc2jit", .module = b.createModule(.{
                    .root_source_file = b.path("bc2jit/lib.zig"),
                    .target = target,
                    .optimize = .ReleaseFast,
                    .link_libc = true,
                    .imports = &.{.{ .name = "pyrlyn_jit", .module = jit_fast }},
                }) },
            },
        }) });
        const tr = target.result;
        const base_path = (b.fmt("bench/baseline-{s}-{s}.txt", .{ @tagName(tr.cpu.arch), @tagName(tr.os.tag) }));
        const check_run = b.addRunArtifact(bt);
        check_run.setEnvironmentVariable("PJIT_BENCH_BASELINE", base_path);
        check_run.has_side_effects = true;
        if (b.option([]const u8, "bench-tol", "bench-test: override the regression tolerance for every case, in percent")) |t| check_run.setEnvironmentVariable("PJIT_BENCH_TOL", t);
        check_run.setCwd(b.path("."));
        b.step("bench-test", "Run benchmark regression tests against the checked-in baseline (-Dbench-tol=PERCENT overrides tolerance)").dependOn(&check_run.step);
        const truncate = b.addSystemCommand(&.{ "sh", "-c", b.fmt("printf '# ns/iter median, zig build bench-update, %s\\n' \"$(date '+%Y-%m-%d') $(uname -m) load: $(uptime | sed 's/.*load averages*: //')\" > '{s}'", .{base_path}) });
        const upd_run = b.addRunArtifact(bt);
        upd_run.setEnvironmentVariable("PJIT_BENCH_BASELINE", base_path);
        upd_run.setEnvironmentVariable("PJIT_BENCH_UPDATE", "1");
        upd_run.has_side_effects = true;
        upd_run.setCwd(b.path("."));
        truncate.setCwd(b.path("."));
        upd_run.step.dependOn(&truncate.step);
        b.step("bench-update", "Re-measure and rewrite the benchmark baseline file").dependOn(&upd_run.step);

        // IR Compiler time/memory (bench/compile_bench.zig), report only; PJIT_BENCH_MS = time budget per case.
        const cbt = b.addExecutable(.{ .name = "compile_bench", .root_module = b.createModule(.{
            .root_source_file = b.path("bench/compile_bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{.{ .name = "pyrlyn_jit", .module = jit_fast }},
        }) });
        const cb_run = b.addRunArtifact(cbt);
        cb_run.has_side_effects = true;
        b.step("bench-compile", "Measure IR Compiler time and memory (x86-64, AArch64, RV64) on generated functions").dependOn(&cb_run.step);
    }

    // zig fmt: our sources only. AsmJit-derived and other generated files are excluded so they
    // stay byte-identical to their generators (see CONTRIBUTING.md).
    const fmt_paths: []const std.Build.LazyPath = &.{ b.path("build.zig"), b.path("src"), b.path("examples"), b.path("bench"), b.path("bc2jit") };
    const fmt_exclude: []const std.Build.LazyPath = &.{
        b.path("third_party"),
        b.path("bench/pre-safety"),
        // AsmJit-derived: ported tests and tables generated from AsmJit's instruction database
        b.path("src/asmjit_x64_tests.zig"),
        b.path("src/asmjit_x86_tests.zig"),
        b.path("src/asmjit_a64_tests.zig"),
        b.path("src/x86_64/isa.zig"),
        b.path("src/aarch64/isa.zig"),
        // generated from LLVM: tables by tools/gen_tables.zig, tests by scripts/gen_*_tests.py
        b.path("src/x86_64/gp_table.zig"),
        b.path("src/x86_64/vec_table.zig"),
        b.path("src/aarch64/a64_table.zig"),
        b.path("src/a64_table_tests.zig"),
        b.path("src/gp_tests.zig"),
        b.path("src/simd_tests.zig"),
        b.path("src/enc_tests.zig"),
        b.path("src/a64_sve_tests.zig"),
        b.path("src/a64_sve_extra_tests.zig"),
    };
    b.step("fmt", "Format sources with zig fmt (generated/AsmJit-derived files excluded)").dependOn(&b.addFmt(.{ .paths = fmt_paths, .exclude_paths = fmt_exclude }).step);
    b.step("fmt-check", "Check formatting with zig fmt --check (same exclusions as fmt)").dependOn(&b.addFmt(.{ .paths = fmt_paths, .exclude_paths = fmt_exclude, .check = true }).step);
}
