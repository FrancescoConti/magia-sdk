#include "tile.h"
#include "rowmax_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Fuses reducemax over the last axis (batched, one lane per row, as reducemax) and the
   element-wise max with the previous row max. */
int rowmax_fp16_spatz_task(void)
{
    const rowmax_fp16_spatz_params_t *params;
    const _Float16 *X;
    const _Float16 *P;
    _Float16 *M;
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t vl;

    params = (const rowmax_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    X    = (const _Float16 *)params->shard_X;
    P    = (const _Float16 *)params->shard_P;
    M    = (_Float16 *)params->shard_M;
    rows = params->dims & 0xffff;
    cols = params->dims >> 16;

    if (MISALIGNED(M) || MISALIGNED(P))
        return 1;

    row_stride = cols * sizeof(_Float16);

    for (size_t r = 0; r < rows; r += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rows - r));

        asm volatile("vlse16.v v0, (%0), %1" ::"r"(X + r * cols), "r"(row_stride));
        for (size_t c = 1; c < cols; c++) {
            asm volatile("vlse16.v v8, (%0), %1" ::"r"(X + r * cols + c), "r"(row_stride));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        if (P) {
            asm volatile("vle16.v v8, (%0)" ::"r"(P + r));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        asm volatile("vse16.v v0, (%0)" ::"r"(M + r) : "memory");
    }

    return 0;
}
