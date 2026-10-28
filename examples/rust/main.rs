//! ZigJIT Rust example: hand-written FFI over include/zigjit.h (no bindgen, no crates).
//! Build: `zig build examples -Drust=true` (or rustc main.rs -L zig-out/lib -l static=zigjit).
use std::ffi::{c_char, c_int, c_void, CStr};

#[repr(C)]
#[derive(Clone, Copy)]
struct A64Op { kind: u8, reg: u8, sub: u8, arr: u8, et: u8, idx: i8, index: u8, index64: u8, ext_op: u8, ext_n: u8, has_ext: u8, _pad: [u8; 5], imm: i64, fimm: f64 }

#[repr(C)]
#[derive(Clone, Copy)]
struct X64Op { kind: u8, reg: u8, size: u8, high: u8, base: u8, index: u8, scale: u8, rip: u8, disp: i32, _pad: u32, imm: i64 }

#[repr(C)] struct Buffer { _p: [u8; 0] }
#[repr(C)] struct Func { _p: [u8; 0] }

#[allow(dead_code)]
extern "C" {
    fn zj_version() -> u32;
    fn zj_status_str(s: c_int) -> *const c_char;
    fn zj_buffer_new(out: *mut *mut Buffer) -> c_int;
    fn zj_buffer_free(b: *mut Buffer);
    fn zj_label_new(b: *mut Buffer, out: *mut u32) -> c_int;
    fn zj_label_bind(b: *mut Buffer, l: u32) -> c_int;
    fn zj_finalize(b: *mut Buffer) -> c_int;
    fn zj_a64_emit(b: *mut Buffer, mn: *const c_char, ops: *const A64Op, n: usize) -> c_int;
    fn zj_a64_branch(b: *mut Buffer, cond: c_int, l: u32) -> c_int;
    fn zj_x64_emit(b: *mut Buffer, mn: *const c_char, ops: *const X64Op, n: usize) -> c_int;
    fn zj_x64_jump(b: *mut Buffer, cc: c_int, l: u32) -> c_int;
    fn zj_jit_add(b: *const Buffer, out: *mut *mut Func) -> c_int;
    fn zj_func_ptr(f: *const Func) -> *const c_void;
    fn zj_func_free(f: *mut Func);
}

/// Owned code buffer (freed on drop).
struct Code(*mut Buffer);
impl Drop for Code { fn drop(&mut self) { unsafe { zj_buffer_free(self.0) } } }

fn ok(s: c_int) -> Result<(), String> {
    if s == 0 { Ok(()) } else { Err(unsafe { CStr::from_ptr(zj_status_str(s)) }.to_string_lossy().into_owned()) }
}


#[allow(dead_code)]
fn x(r: u8) -> A64Op { A64Op { kind: 0, reg: r, sub: 1, arr: 0, et: 0, idx: -1, index: 0xFF, index64: 0, ext_op: 0, ext_n: 0, has_ext: 0, _pad: [0; 5], imm: 0, fimm: 0.0 } }

#[allow(dead_code)]
fn r64(r: u8) -> X64Op { X64Op { kind: 0, reg: r, size: 8, high: 0, base: 0xFF, index: 0xFF, scale: 0, rip: 0, disp: 0, _pad: 0, imm: 0 } }

fn run() -> Result<i64, String> {
    unsafe {
        let mut raw = std::ptr::null_mut();
        ok(zj_buffer_new(&mut raw))?;
        let b = Code(raw);
        let mut done = 0u32;
        ok(zj_label_new(b.0, &mut done))?;
        #[cfg(target_arch = "aarch64")]
        {
            ok(zj_a64_emit(b.0, c"add".as_ptr(), [x(0), x(0), x(1)].as_ptr(), 3))?;
            ok(zj_a64_branch(b.0, -1, done))?;
            let mut i = x(0); i.kind = 2; i.imm = 1;
            ok(zj_a64_emit(b.0, c"brk".as_ptr(), &i, 1))?;
            ok(zj_label_bind(b.0, done))?;
            ok(zj_a64_emit(b.0, c"ret".as_ptr(), std::ptr::null(), 0))?;
        }
        #[cfg(target_arch = "x86_64")]
        {
            ok(zj_x64_emit(b.0, c"mov".as_ptr(), [r64(0), r64(7)].as_ptr(), 2))?;
            ok(zj_x64_emit(b.0, c"add".as_ptr(), [r64(0), r64(6)].as_ptr(), 2))?;
            ok(zj_x64_jump(b.0, -1, done))?;
            ok(zj_x64_emit(b.0, c"int3".as_ptr(), std::ptr::null(), 0))?;
            ok(zj_label_bind(b.0, done))?;
            ok(zj_x64_emit(b.0, c"ret".as_ptr(), std::ptr::null(), 0))?;
        }
        ok(zj_finalize(b.0))?;
        let mut f = std::ptr::null_mut();
        ok(zj_jit_add(b.0, &mut f))?;
        let func: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(zj_func_ptr(f));
        let r = func(40, 2);
        zj_func_free(f);
        Ok(r)
    }
}

fn main() {
    match run() {
        Ok(r) => {
            println!("zigjit {:#08x} (Rust): add(40, 2) = {}", unsafe { zj_version() }, r);
            std::process::exit(if r == 42 { 0 } else { 1 });
        }
        Err(e) => { eprintln!("error: {e}"); std::process::exit(1) }
    }
}
