#include "tile.h"
#include "rowdiv_fp16_spatz_params.h"

/**
 * Divides one row by a scalar: dst[c] = src[c] / s.
 */
static inline void div_row(const _Float16 *src, _Float16 *dst, size_t len, _Float16 s)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl   = len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));

        asm volatile("vfdiv.vf v0, v0, %0" ::"f"(s));

        asm volatile("vse16.v v0, (%0)" ::"r"(p_dst) : "memory");

        p_src += vl;
        p_dst += vl;
    }
}

int rowdiv_fp16_spatz_task(void)
{
    volatile rowdiv_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *X;
    const _Float16 *S;
    _Float16 *Y;
    size_t cols;
    size_t rows;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile rowdiv_fp16_spatz_params_t *)params_addr;

    X    = (const _Float16 *)params->shard_X;
    S    = (const _Float16 *)params->shard_S;
    Y    = (_Float16 *)params->shard_Y;
    cols = params->cols;
    rows = params->row_len;

    for (size_t r = 0; r < rows; r++)
        div_row(X + r * cols, Y + r * cols, cols, S[r]);

    return 0;
}
