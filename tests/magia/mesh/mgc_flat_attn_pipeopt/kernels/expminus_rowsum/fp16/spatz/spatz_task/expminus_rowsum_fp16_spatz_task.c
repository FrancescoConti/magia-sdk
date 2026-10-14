#include "tile.h"
#include "expminus_rowsum_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Fuses expminus (with the scale folded into the exp), reducesum over the last axis and
   exp of the max difference, with the same arithmetic as the separate kernels.
   exp(s * x) uses COEF * s and the clamps divided by s, precomputed by the caller: exact if s
   is a power of 2. */

/* v0 = reinterpret_fp16(coef * max(v0, min) + BIAS). */
static inline void fast_exp_v0(_Float16 min, _Float16 coef)
{
    register _Float16 bias asm("f11") = EXPMINUS_ROWSUM_EXP_BIAS;

    asm volatile("vfmax.vf v0, v0, %0" ::"f"(min));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(coef));
    asm volatile("vfadd.vf v0, v0, %0" ::"f"(bias));
    asm volatile("vfcvt.rtz.xu.f.v v0, v0");
}

/* One row, vectorized along cols. A misaligned row (odd cols) peels its first element. */
static inline void expminus_row(
    const _Float16 *src, _Float16 *dst, size_t len, _Float16 m, _Float16 min, _Float16 coef)
{
    _Float16 first;
    size_t vl;

    if (MISALIGNED(src)) {
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(1));
        asm volatile("vfmv.s.f v0, %0" ::"f"(*src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
        fast_exp_v0(min, coef);
        asm volatile("vfmv.f.s %0, v0" : "=f"(first));
        *dst++ = first;
        src++;
        len--;
    }

    for (; len > 0; len -= vl, src += vl, dst += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(len));
        asm volatile("vle16.v v0, (%0)" ::"r"(src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
        fast_exp_v0(min, coef);
        asm volatile("vse16.v v0, (%0)" ::"r"(dst) : "memory");
    }
}

/* S[r] = sum_c Y[r, c], one lane per row, fp32 accumulation (as reducesum). */
static inline void rowsum(const _Float16 *Y, _Float16 *S, size_t rows, size_t cols)
{
    register _Float16 zero asm("fs0") = 0.0f;
    const size_t row_stride = cols * sizeof(_Float16);
    size_t vl;

    for (size_t r = 0; r < rows; r += vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(rows - r));

        asm volatile("vlse16.v v8, (%0), %1" ::"r"(Y + r * cols), "r"(row_stride));
        asm volatile("vfwadd.vf v0, v8, %0" ::"f"(zero));
        for (size_t c = 1; c < cols; c++) {
            asm volatile("vlse16.v v8, (%0), %1" ::"r"(Y + r * cols + c), "r"(row_stride));
            asm volatile("vfwadd.wv v0, v0, v8");
        }

        asm volatile("vfncvt.f.f.w v8, v0");
        asm volatile("vse16.v v8, (%0)" ::"r"(S + r) : "memory");
    }
}

/* A[r] = exp(s * (A[r] - M[r])), with the clamps of the exp kernel. */
static inline void exp_diff(_Float16 *A, const _Float16 *M, size_t rows, _Float16 min,
                            _Float16 max, _Float16 coef)
{
    size_t vl;

    for (size_t r = 0; r < rows; r += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rows - r));
        asm volatile("vle16.v v0, (%0)" ::"r"(A + r));
        asm volatile("vle16.v v8, (%0)" ::"r"(M + r));
        asm volatile("vfsub.vv v0, v0, v8");
        asm volatile("vfmin.vf v0, v0, %0" ::"f"(max));
        fast_exp_v0(min, coef);
        asm volatile("vse16.v v0, (%0)" ::"r"(A + r) : "memory");
    }
}

int expminus_rowsum_fp16_spatz_task(void)
{
    const expminus_rowsum_fp16_spatz_params_t *params;
    const _Float16 *X;
    const _Float16 *M;
    _Float16 *Y;
    _Float16 *S;
    _Float16 *A;
    union { uint32_t u; _Float16 f[2]; } min_max, coef;
    _Float16 min, max;
    size_t rows;
    size_t cols;

    params = (const expminus_rowsum_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    X          = (const _Float16 *)params->shard_X;
    M          = (const _Float16 *)params->shard_M;
    Y          = (_Float16 *)params->shard_Y;
    S          = (_Float16 *)params->shard_S;
    A          = (_Float16 *)params->shard_A;
    rows       = params->dims & 0xffff;
    cols       = params->dims >> 16;
    min_max.u  = params->min_max;
    coef.u     = params->coef;
    min        = min_max.f[0];
    max        = min_max.f[1];

    if (rows == 0 || cols == 0)
        return 0;

    /* Peeling aligns X and Y together only if they share alignment. */
    if (MISALIGNED((uintptr_t)X ^ (uintptr_t)Y) || MISALIGNED(S) || MISALIGNED(A) ||
        MISALIGNED(M))
        return 1;

    for (size_t r = 0; r < rows; r++)
        expminus_row(X + r * cols, Y + r * cols, cols, M[r], min, coef.f[0]);

    rowsum(Y, S, rows, cols);

    if (A)
        exp_diff(A, M, rows, min, max, coef.f[0]);

    return 0;
}
