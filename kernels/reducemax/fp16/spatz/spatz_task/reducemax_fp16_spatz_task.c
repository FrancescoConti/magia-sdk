#include "tile.h"
#include "reducemax_fp16_spatz_params.h"

/* Unit-stride vector accesses must start 4-byte aligned. With an odd row length, every other
   row starts at a 2 mod 4 address: those go through a strided access with a one-element
   stride, which the VLSU splits into single-element requests. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)
#define ELEM_STRIDE    (sizeof(_Float16))

/* inner_dim > 1: the reduced elements are inner_dim apart, so vectorize across inner_dim and
   take the element-wise max of the reduce_dim rows. */
static inline void reduce_max_strided(const _Float16 *src,
                                      _Float16 *dst,
                                      const size_t reduce_dim,
                                      const size_t inner_dim)
{
    size_t avl = inner_dim;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        /* Seed with the first row, not with 0, so all-negative inputs are handled. */
        const _Float16 *p_first = src + (inner_dim - avl);
        if (MISALIGNED(p_first))
            asm volatile("vlse16.v v0, (%0), %1" ::"r"(p_first), "r"(ELEM_STRIDE));
        else
            asm volatile("vle16.v v0, (%0)" ::"r"(p_first));

        for (size_t r = 1; r < reduce_dim; r++) {
            const _Float16 *p_src = src + (r * inner_dim) + (inner_dim - avl);

            if (MISALIGNED(p_src))
                asm volatile("vlse16.v v8, (%0), %1" ::"r"(p_src), "r"(ELEM_STRIDE));
            else
                asm volatile("vle16.v v8, (%0)" ::"r"(p_src));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        _Float16 *p_dst = dst + (inner_dim - avl);
        if (MISALIGNED(p_dst))
            asm volatile("vsse16.v v0, (%0), %1" ::"r"(p_dst), "r"(ELEM_STRIDE) : "memory");
        else
            asm volatile("vse16.v v0, (%0)" ::"r"(p_dst) : "memory");
    }
}

/* inner_dim == 1: the reduce_dim elements are contiguous, so vectorize along them and fold
   each chunk into a scalar accumulator held in v16[0]. */
static inline void reduce_max_contiguous(const _Float16 *src, _Float16 *dst, const size_t reduce_dim)
{
    size_t avl = reduce_dim;
    size_t vl;
    _Float16 max;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vfmv.s.f v16, %0" ::"f"(src[0]));

    /* src[0] is already in the accumulator: skipping it realigns a 2 mod 4 row start. */
    if (MISALIGNED(src)) {
        src++;
        avl--;
    }

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vle16.v v8, (%0)" ::"r"(src));
        asm volatile("vfredmax.vs v16, v8, v16");

        src += vl;
    }

    asm volatile("vfmv.f.s %0, v16" : "=f"(max));
    *dst = max;
}

int reducemax_fp16_spatz_task(void)
{
    volatile reducemax_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    const _Float16 *src;
    _Float16 *dst;
    uint32_t reduce_dim;
    uint32_t inner_dim;
    uint32_t outer_len;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile reducemax_fp16_spatz_params_t *)params_addr;

    src        = (_Float16 *)params->shard_X;
    dst        = (_Float16 *)params->shard_Y;
    reduce_dim = params->reduce_dim;
    inner_dim  = params->inner_dim;
    outer_len  = params->outer_len;

    uint32_t in_stride  = reduce_dim * inner_dim;
    uint32_t out_stride = inner_dim;

    for (uint32_t o = 0; o < outer_len; o++) {
        if (inner_dim == 1)
            reduce_max_contiguous(src + (o * in_stride), dst + o, reduce_dim);
        else
            reduce_max_strided(src + (o * in_stride), dst + (o * out_stride), reduce_dim, inner_dim);
    }

    return 0;
}
