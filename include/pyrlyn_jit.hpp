// Header-only C++17 RAII wrapper over pyrlyn_jit.h. Errors are thrown as pyrlyn_jit::Error.
#pragma once
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pyrlyn_jit.h"

namespace pyrlyn_jit {

struct Error : std::runtime_error {
    int status;
    Error(int s, const char *what) : std::runtime_error(std::string(what) + ": " + pjit_status_str(s)), status(s) {}
};

inline void check(int s, const char *what) {
    if (s != PJIT_OK) throw Error(s, what);
}

class Function;

class Buffer {
public:
    Buffer() { check(pjit_buffer_new(&b_), "pjit_buffer_new"); }
    ~Buffer() { pjit_buffer_free(b_); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&o) noexcept : b_(std::exchange(o.b_, nullptr)) {}
    Buffer &operator=(Buffer &&o) noexcept { std::swap(b_, o.b_); return *this; }

    uint32_t newLabel() { uint32_t l; check(pjit_label_new(b_, &l), "pjit_label_new"); return l; }
    void bind(uint32_t l) { check(pjit_label_bind(b_, l), "pjit_label_bind"); }
    void finalize() { check(pjit_finalize(b_), "pjit_finalize"); }
    void reset() { pjit_buffer_reset(b_); }

    void a64(const char *mn, std::initializer_list<pjit_a64_op> ops) { check(pjit_a64_emit(b_, mn, ops.begin(), ops.size()), mn); }
    void a64Branch(int cond, uint32_t l) { check(pjit_a64_branch(b_, cond, l), "pjit_a64_branch"); }
    void x64(const char *mn, std::initializer_list<pjit_x64_op> ops) { check(pjit_x64_emit(b_, mn, ops.begin(), ops.size()), mn); }
    void x64Jump(int cc, uint32_t l) { check(pjit_x64_jump(b_, cc, l), "pjit_x64_jump"); }

    std::vector<uint8_t> code() const {
        size_t n = 0;
        const uint8_t *p = pjit_buffer_code(b_, &n);
        return std::vector<uint8_t>(p, p + n);
    }
    pjit_buffer *raw() const { return b_; }

private:
    pjit_buffer *b_ = nullptr;
};

class Function {
public:
    explicit Function(const Buffer &b) { check(pjit_jit_add(b.raw(), &f_), "pjit_jit_add"); }
    ~Function() { pjit_func_free(f_); }
    Function(const Function &) = delete;
    Function &operator=(const Function &) = delete;
    Function(Function &&o) noexcept : f_(std::exchange(o.f_, nullptr)) {}
    template <class F> F as() const { return reinterpret_cast<F>(const_cast<void *>(pjit_func_ptr(f_))); }

private:
    pjit_func *f_ = nullptr;
};

} // namespace pyrlyn_jit
