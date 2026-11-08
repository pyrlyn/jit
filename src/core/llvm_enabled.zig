//! LLVM C API implementation, imported only by LLVM-enabled builds.
const std = @import("std");
const ir = @import("ir.zig");
const api = @import("llvm.zig");
const c = @import("llvm_c");

const Fn = *const fn (i64, i64, i64, i64) callconv(.c) i64;
pub const Program = struct {
    context: c.LLVMContextRef,
    module: c.LLVMModuleRef, // owned by engine
    engine: c.LLVMExecutionEngineRef,
    function: c.LLVMValueRef,
    native: ?Fn,

    pub fn deinit(p: *Program) void {
        c.LLVMDisposeExecutionEngine(p.engine);
        c.LLVMContextDispose(p.context);
        p.* = undefined;
    }
    /// LLVM's interpreter is not reentrant; serialize calls on a single Program.
    pub fn call(p: *const Program, args: [4]i64) i64 {
        if (p.native) |f| return f(args[0], args[1], args[2], args[3]);
        var values: [4]c.LLVMGenericValueRef = undefined;
        for (args, 0..) |arg, i| values[i] = c.LLVMCreateGenericValueOfInt(c.LLVMInt64TypeInContext(p.context), @bitCast(arg), 1);
        defer for (values) |v| c.LLVMDisposeGenericValue(v);
        const result = c.LLVMRunFunction(p.engine, p.function, 4, &values);
        defer c.LLVMDisposeGenericValue(result);
        return @bitCast(c.LLVMGenericValueToInt(result, 1));
    }
    /// Caller frees the LLVM IR snapshot with gpa.free.
    pub fn irText(p: *const Program, gpa: std.mem.Allocator) api.Error![]u8 {
        const text = c.LLVMPrintModuleToString(p.module);
        defer c.LLVMDisposeMessage(text);
        return gpa.dupe(u8, std.mem.span(text));
    }
};

// LLVM target registration is process-global, unlike contexts/modules/builders.
var initialized = std.atomic.Value(u8).init(0);
fn initialize() api.Error!void {
    if (initialized.cmpxchgStrong(0, 1, .acq_rel, .acquire) == null) {
        c.LLVMLinkInMCJIT();
        c.LLVMLinkInInterpreter();
        const failed = c.LLVMInitializeNativeTarget() != 0 or c.LLVMInitializeNativeAsmPrinter() != 0;
        initialized.store(if (failed) 3 else 2, .release);
    }
    while (initialized.load(.acquire) == 1) std.atomic.spinLoopHint();
    if (initialized.load(.acquire) != 2) return error.LlvmFailed;
}

fn verify(module: c.LLVMModuleRef, diag: *api.Diagnostic) api.Error!void {
    var msg: [*c]u8 = null;
    const failed = c.LLVMVerifyModule(module, c.LLVMReturnStatusAction, &msg) != 0;
    defer if (msg != null) c.LLVMDisposeMessage(msg);
    if (failed) {
        if (msg != null) diag.set(std.mem.span(msg));
        return error.InvalidIr;
    }
}

pub fn prepare(gpa: std.mem.Allocator, f: *const ir.Func, options: api.Options, diag: *api.Diagnostic) api.Error!Program {
    f.verify() catch return error.InvalidIr;
    if (f.params > 4) return error.UnsupportedIr;
    for (f.blocks.items) |block| {
        if (block.params.len != 0) return error.UnsupportedIr;
        for (block.ops.items) |op| switch (op) {
            .@"const", .copy, .arith, .br, .cbr, .ret => {},
            else => return error.UnsupportedIr,
        };
    }
    try initialize();
    const ctx = c.LLVMContextCreate();
    errdefer c.LLVMContextDispose(ctx);
    const module = c.LLVMModuleCreateWithNameInContext("kxbc", ctx);
    var owns_module = true;
    defer if (owns_module) c.LLVMDisposeModule(module);
    const triple = c.LLVMGetDefaultTargetTriple();
    defer c.LLVMDisposeMessage(triple);
    c.LLVMSetTarget(module, triple);
    const builder = c.LLVMCreateBuilderInContext(ctx);
    defer c.LLVMDisposeBuilder(builder);
    const i64_ty = c.LLVMInt64TypeInContext(ctx);
    var param_types: [4]c.LLVMTypeRef = @splat(i64_ty);
    const fn_ty = c.LLVMFunctionType(i64_ty, &param_types, 4, 0);
    const function = c.LLVMAddFunction(module, "kxbc_entry", fn_ty);
    const entry = c.LLVMAppendBasicBlockInContext(ctx, function, "entry");
    const blocks = try gpa.alloc(c.LLVMBasicBlockRef, f.blocks.items.len);
    defer gpa.free(blocks);
    for (blocks) |*block| block.* = c.LLVMAppendBasicBlockInContext(ctx, function, "bb");
    c.LLVMPositionBuilderAtEnd(builder, entry);
    const slots = try gpa.alloc(c.LLVMValueRef, f.num_vregs);
    defer gpa.free(slots);
    for (slots, 0..) |*slot, i| {
        slot.* = c.LLVMBuildAlloca(builder, i64_ty, "reg");
        const initial = if (i < f.params) c.LLVMGetParam(function, @intCast(i)) else c.LLVMConstInt(i64_ty, 0, 0);
        _ = c.LLVMBuildStore(builder, initial, slot.*);
    }
    _ = c.LLVMBuildBr(builder, blocks[0]);
    const emitter: Emitter = .{ .builder = builder, .ty = i64_ty, .slots = slots };
    for (f.blocks.items, blocks) |block, llvm_block| {
        c.LLVMPositionBuilderAtEnd(builder, llvm_block);
        for (block.ops.items) |op| switch (op) {
            .@"const" => |x| emitter.store(x.dst, c.LLVMConstInt(i64_ty, @bitCast(x.imm), 1)),
            .copy => |x| emitter.store(x.dst, emitter.load(x.src)),
            .arith => |x| {
                const a = emitter.load(x.a);
                var b = emitter.load(x.b);
                // LLVM shifts are poison for counts >=64; KXBC and our IR mask them.
                if (x.op == .shl or x.op == .shr or x.op == .sar)
                    b = c.LLVMBuildAnd(builder, b, c.LLVMConstInt(i64_ty, 63, 0), "shift");
                const value = switch (x.op) {
                    .add => c.LLVMBuildAdd(builder, a, b, "add"),
                    .sub => c.LLVMBuildSub(builder, a, b, "sub"),
                    .mul => c.LLVMBuildMul(builder, a, b, "mul"),
                    .@"and" => c.LLVMBuildAnd(builder, a, b, "and"),
                    .@"or" => c.LLVMBuildOr(builder, a, b, "or"),
                    .xor => c.LLVMBuildXor(builder, a, b, "xor"),
                    .shl => c.LLVMBuildShl(builder, a, b, "shl"),
                    .shr => c.LLVMBuildLShr(builder, a, b, "shr"),
                    .sar => c.LLVMBuildAShr(builder, a, b, "sar"),
                    // KXBC lowering already guards zero and -1 divisors.
                    .sdiv => c.LLVMBuildSDiv(builder, a, b, "div"),
                    .srem => c.LLVMBuildSRem(builder, a, b, "rem"),
                };
                emitter.store(x.dst, value);
            },
            .br => |target| {
                _ = c.LLVMBuildBr(builder, blocks[target]);
            },
            .cbr => |x| {
                const pred: c.LLVMIntPredicate = switch (x.cond) {
                    .eq => c.LLVMIntEQ,
                    .ne => c.LLVMIntNE,
                    .lt => c.LLVMIntSLT,
                    .le => c.LLVMIntSLE,
                    .gt => c.LLVMIntSGT,
                    .ge => c.LLVMIntSGE,
                    .ult => c.LLVMIntULT,
                    .ule => c.LLVMIntULE,
                    .ugt => c.LLVMIntUGT,
                    .uge => c.LLVMIntUGE,
                };
                const cond = c.LLVMBuildICmp(builder, pred, emitter.load(x.a), emitter.load(x.b), "cond");
                _ = c.LLVMBuildCondBr(builder, cond, blocks[x.then], blocks[x.@"else"]);
            },
            .ret => |v| {
                _ = c.LLVMBuildRet(builder, emitter.load(v));
            },
            else => unreachable,
        };
    }
    try verify(module, diag);
    if (options.optimize) {
        const passes = c.LLVMCreatePassBuilderOptions();
        defer c.LLVMDisposePassBuilderOptions(passes);
        c.LLVMPassBuilderOptionsSetVerifyEach(passes, 1);
        // Deliberately scalar: the same IR is supported by LLVM Interpreter and MCJIT.
        if (c.LLVMRunPasses(module, "function(mem2reg,sccp,adce,simplifycfg)", null, passes)) |err| {
            const msg = c.LLVMGetErrorMessage(err);
            defer c.LLVMDisposeErrorMessage(msg);
            diag.set(std.mem.span(msg));
            return error.LlvmFailed;
        }
    }
    try verify(module, diag);
    var engine: c.LLVMExecutionEngineRef = null;
    var msg: [*c]u8 = null;
    defer if (msg != null) c.LLVMDisposeMessage(msg);
    // EngineBuilder takes the module even when engine creation fails.
    owns_module = false;
    const failed = switch (options.mode) {
        .interpreter => c.LLVMCreateInterpreterForModule(&engine, module, &msg),
        .jit => c.LLVMCreateJITCompilerForModule(&engine, module, if (options.optimize) 2 else 0, &msg),
    };
    if (failed != 0) {
        if (msg != null) diag.set(std.mem.span(msg));
        return error.LlvmFailed;
    }
    errdefer c.LLVMDisposeExecutionEngine(engine);
    var native: ?Fn = null;
    if (options.mode == .jit) {
        const address = c.LLVMGetFunctionAddress(engine, "kxbc_entry");
        if (address == 0) {
            diag.set("LLVM did not produce kxbc_entry");
            return error.LlvmFailed;
        }
        native = @ptrFromInt(address);
    }
    return .{ .context = ctx, .module = module, .engine = engine, .function = function, .native = native };
}

const Emitter = struct {
    builder: c.LLVMBuilderRef,
    ty: c.LLVMTypeRef,
    slots: []const c.LLVMValueRef,
    fn load(e: Emitter, v: ir.VReg) c.LLVMValueRef {
        return c.LLVMBuildLoad2(e.builder, e.ty, e.slots[v], "value");
    }
    fn store(e: Emitter, v: ir.VReg, value: c.LLVMValueRef) void {
        _ = c.LLVMBuildStore(e.builder, value, e.slots[v]);
    }
};
