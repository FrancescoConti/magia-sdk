// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (synchronously, on the CV32 control core) by the
// mgc-generated test.c. All pointer arguments are L1 addresses.
// The softmax uses the merged kernels in ../kernels instead of the generic ones.

#include <stddef.h>
#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "add_fp16_spatz.h"
#include "colsum_rescale_fp16_spatz.h"
#include "expminus_rowsum_fp16_spatz.h"
#include "memset_fp16_spatz.h"
#include "reducemax_fp16_spatz.h"
#include "rowdiv_fp16_spatz.h"
#include "rowmax_fp16_spatz.h"
#include "rowscale_fp16_spatz.h"
#include "eventunit.h"
#include "mgc_flat_attn_pipeopt_spatz_task_bin.h"

/* Per-tile flag: Spatz is set up on each tile before its first task. */
static volatile uint8_t spatz_ready __attribute__((section(".tile_bss")));

/**
 * Lazy Spatz setup on this tile (event unit line, binary and clock): the mgc-generated main
 * does not know about Spatz.
 */
static void spatz_setup(void)
{
    if (spatz_ready)
        return;
    eu_spatz_init(0);
    spatz_init(SPATZ_BINARY_START);
    spatz_ready = 1;
}

/**
 * y[k] = 0 for `n` fp16 elements, on this tile's Spatz.
 */
void l1_memset_zero_fp16_spatz(uint32_t y, uint32_t n)
{
    spatz_setup();
    MAGIA_memset_zero_fp16_spatz_l1((float16 *)y, n);
}

/**
 * m[r] = max_c x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_rowmax_fp16_spatz(uint32_t m, uint32_t x, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_rowmax_fp16_spatz_l1((const float16 *)x, NULL, (float16 *)m, rows, cols);
}

/**
 * m[r] = max(max_c x[r, c], p[r]) for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_rowmax_max_fp16_spatz(uint32_t m, uint32_t x, uint32_t p, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_rowmax_fp16_spatz_l1((const float16 *)x, (const float16 *)p, (float16 *)m, rows, cols);
}

/**
 * m[c] = max_r x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz
 * (vectorized along the columns).
 */
void l1_colmax_fp16_spatz(uint32_t m, uint32_t x, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_reducemax_fp16_spatz_l1((const float16 *)x, (float16 *)m, 1, rows, cols);
}

/**
 * y[r, c] = exp((x[r, c] - m[r]) / sqrt(PROJ_SIZE)) and s[r] = sum_c y[r, c] for a `rows` x
 * `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_expminus_rowsum_fp16_spatz(
    uint32_t y, uint32_t s, uint32_t x, uint32_t m, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_expminus_rowsum_fp16_spatz_l1((const float16 *)x,
                                        (const float16 *)m,
                                        (float16 *)y,
                                        (float16 *)s,
                                        NULL,
                                        (float16)(1.0f / __builtin_sqrtf((float)PROJ_SIZE)),
                                        rows,
                                        cols);
}

/**
 * As l1_expminus_rowsum_fp16_spatz, plus a[r] = exp((a[r] - m[r]) / sqrt(PROJ_SIZE)) in place.
 */
void l1_expminus_rowsum_alpha_fp16_spatz(
    uint32_t y, uint32_t s, uint32_t a, uint32_t x, uint32_t m, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_expminus_rowsum_fp16_spatz_l1((const float16 *)x,
                                        (const float16 *)m,
                                        (float16 *)y,
                                        (float16 *)s,
                                        (float16 *)a,
                                        (float16)(1.0f / __builtin_sqrtf((float)PROJ_SIZE)),
                                        rows,
                                        cols);
}

/**
 * s[c] += sum_{r >= 1} g[r, c] for a `rows` x `cols` fp16 matrix g, on this tile's Spatz.
 */
void l1_colsum_acc_fp16_spatz(uint32_t s, uint32_t g, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_colsum_rescale_fp16_spatz_l1((float16 *)s, (const float16 *)g, NULL, NULL, rows, cols);
}

/**
 * As l1_colsum_acc_fp16_spatz, then p[c] = a[c] * p[c] and s[c] = p[c] + s[c].
 */
void l1_colsum_acc_rescale_fp16_spatz(
    uint32_t s, uint32_t g, uint32_t p, uint32_t a, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_colsum_rescale_fp16_spatz_l1(
        (float16 *)s, (const float16 *)g, (float16 *)p, (const float16 *)a, rows, cols);
}

/**
 * y[k] = a[k] + b[k] for `n` fp16 elements, on this tile's Spatz.
 */
void l1_add_fp16_spatz(uint32_t y, uint32_t a, uint32_t b, uint32_t n)
{
    spatz_setup();
    MAGIA_add_fp16_spatz_l1((const float16 *)a, (const float16 *)b, (float16 *)y, n);
}

/**
 * y[r, c] = x[r, c] * s[r] for a `rows` x `cols` fp16 matrix, i.e. y = diag(s) x, on this
 * tile's Spatz.
 */
void l1_rowscale_fp16_spatz(uint32_t y, uint32_t x, uint32_t s, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_rowscale_fp16_spatz_l1((const float16 *)x, (const float16 *)s, (float16 *)y, rows, cols);
}

/**
 * y[r, c] = x[r, c] / s[r] for a `rows` x `cols` fp16 matrix, i.e. y = diag(s)^-1 x, on this
 * tile's Spatz.
 */
void l1_rowdiv_fp16_spatz(uint32_t y, uint32_t x, uint32_t s, uint32_t rows, uint32_t cols)
{
    spatz_setup();
    MAGIA_rowdiv_fp16_spatz_l1((const float16 *)x, (const float16 *)s, (float16 *)y, rows, cols);
}
