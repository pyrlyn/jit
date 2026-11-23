//! Cranelift (cranelift-jit) side of the benchmark: sum_i64(ptr, n) and fact_u64(n).
use cranelift_codegen::ir::{condcodes::IntCC, types::I64, AbiParam, InstBuilder, MemFlagsData, Signature, UserFuncName, Function};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{control::ControlPlane, isa::OwnedTargetIsa, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::time::Instant;
use cranelift_codegen::isa::TargetFrontendConfig;

fn isa() -> OwnedTargetIsa {
    let mut f = settings::builder();
    f.set("opt_level", "speed").unwrap();
    cranelift_native::builder().unwrap().finish(settings::Flags::new(f)).unwrap()
}

fn sig(isa: &OwnedTargetIsa, nparams: usize) -> Signature {
    let mut s = Signature::new(isa.default_call_conv());
    for _ in 0..nparams { s.params.push(AbiParam::new(I64)); }
    s.returns.push(AbiParam::new(I64));
    s
}

// sum(p, n): end = p + n*8; s = 0; while p <u end { s += *p; p += 8 } return s
fn build_sum(f: &mut Function, fbc: &mut FunctionBuilderContext, fc: TargetFrontendConfig) {
    let mut b = FunctionBuilder::new(f, fbc);
    let (e, lp, body, done) = (b.create_block(), b.create_block(), b.create_block(), b.create_block());
    b.append_block_params_for_function_params(e);
    b.append_block_param(lp, I64); b.append_block_param(lp, I64);
    b.append_block_param(done, I64);
    b.switch_to_block(e);
    let (p0, n) = (b.block_params(e)[0], b.block_params(e)[1]);
    let off = b.ins().ishl_imm(n, 3);
    let end = b.ins().iadd(p0, off);
    let z = b.ins().iconst(I64, 0);
    b.ins().jump(lp, &[p0.into(), z.into()]);
    b.switch_to_block(lp);
    let (p, s) = (b.block_params(lp)[0], b.block_params(lp)[1]);
    let c = b.ins().icmp(IntCC::UnsignedLessThan, p, end);
    b.ins().brif(c, body, &[], done, &[s.into()]);
    b.switch_to_block(body);
    let x = b.ins().load(I64, MemFlagsData::trusted(), p, 0);
    let s2 = b.ins().iadd(s, x);
    let p2 = b.ins().iadd_imm(p, 8);
    b.ins().jump(lp, &[p2.into(), s2.into()]);
    b.switch_to_block(done);
    let r = b.block_params(done)[0];
    b.ins().return_(&[r]);
    b.seal_all_blocks();
    b.finalize(fc);
}

// fact(n): r = 1; i = 1; while i <=u n { r *= i; i += 1 } return r   (wrapping)
fn build_fact(f: &mut Function, fbc: &mut FunctionBuilderContext, fc: TargetFrontendConfig) {
    let mut b = FunctionBuilder::new(f, fbc);
    let (e, lp, body, done) = (b.create_block(), b.create_block(), b.create_block(), b.create_block());
    b.append_block_params_for_function_params(e);
    b.append_block_param(lp, I64); b.append_block_param(lp, I64);
    b.append_block_param(done, I64);
    b.switch_to_block(e);
    let n = b.block_params(e)[0];
    let one = b.ins().iconst(I64, 1);
    b.ins().jump(lp, &[one.into(), one.into()]);
    b.switch_to_block(lp);
    let (i, r) = (b.block_params(lp)[0], b.block_params(lp)[1]);
    let c = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, i, n);
    b.ins().brif(c, body, &[], done, &[r.into()]);
    b.switch_to_block(body);
    let r2 = b.ins().imul(r, i);
    let i2 = b.ins().iadd_imm(i, 1);
    b.ins().jump(lp, &[i2.into(), r2.into()]);
    b.switch_to_block(done);
    let r = b.block_params(done)[0];
    b.ins().return_(&[r]);
    b.seal_all_blocks();
    b.finalize(fc);
}

fn median(v: &mut Vec<u64>) -> u64 { v.sort(); v[v.len() / 2] }

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or("out".into());
    let isa = isa();
    let builders: [(&str, usize, fn(&mut Function, &mut FunctionBuilderContext, TargetFrontendConfig)); 2] =
        [("sum", 2, build_sum), ("fact", 1, build_fact)];
    let mut fbc = FunctionBuilderContext::new();
    let mut ctx = Context::new();
    let mut jm = JITModule::new(JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names()));
    let mut ptrs = Vec::new();
    for (name, np, build) in builders {
        // (1) IR build + codegen to bytes (reused Context, like a real JIT loop)
        let mut ts = Vec::new();
        let mut code = Vec::new();
        for _ in 0..1000 {
            let t0 = Instant::now();
            ctx.clear();
            ctx.func = Function::with_name_signature(UserFuncName::user(0, 0), sig(&isa, np));
            build(&mut ctx.func, &mut fbc, isa.frontend_config());
            let cc = ctx.compile(isa.as_ref(), &mut ControlPlane::default()).unwrap();
            let len = cc.code_buffer().len();
            ts.push(t0.elapsed().as_nanos() as u64);
            if code.is_empty() { code = cc.code_buffer().to_vec(); }
            std::hint::black_box(len);
        }
        std::fs::write(format!("{out_dir}/cranelift_{name}.bin"), &code).unwrap();
        // (2) full cranelift-jit path: IR build + declare/define + finalize_definitions (exec memory)
        let mut tf = Vec::new();
        let mut last = std::ptr::null();
        for k in 0..1000 {
            let t0 = Instant::now();
            let id = jm.declare_function(&format!("{name}{k}"), Linkage::Local, &sig(&isa, np)).unwrap();
            ctx.clear();
            ctx.func = Function::with_name_signature(UserFuncName::user(0, k), sig(&isa, np));
            build(&mut ctx.func, &mut fbc, isa.frontend_config());
            jm.define_function(id, &mut ctx).unwrap();
            jm.finalize_definitions().unwrap();
            last = jm.get_finalized_function(id);
            tf.push(t0.elapsed().as_nanos() as u64);
        }
        ptrs.push(last);
        println!("cranelift {name} compile_ns={} full_ns={} bytes={}", median(&mut ts), median(&mut tf), code.len());
    }
    let sum: extern "C" fn(*const i64, i64) -> i64 = unsafe { std::mem::transmute(ptrs[0]) };
    let fact: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptrs[1]) };
    let n = 10_000_000usize;
    let data: Vec<i64> = (0..n as i64).map(|i| i.wrapping_mul(2654435761) ^ (i >> 3)).collect();
    let mut te = Vec::new(); let mut res = 0;
    for _ in 0..21 { let t0 = Instant::now(); res = sum(data.as_ptr(), n as i64); te.push(t0.elapsed().as_nanos() as u64); }
    println!("cranelift sum exec_ns={} result={}", median(&mut te), res);
    let mut te = Vec::new(); let mut acc = 0i64;
    for _ in 0..21 {
        let t0 = Instant::now(); acc = 0;
        for k in 0..1_000_000 { acc = acc.wrapping_add(fact(std::hint::black_box(20 + (k & 1)))); }
        te.push(t0.elapsed().as_nanos() as u64);
    }
    println!("cranelift fact exec_ns={} result={}", median(&mut te), acc);
}
