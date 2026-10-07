// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"

#include "kernels_compare_utils.h"
#include "reducemax_fp16_spatz.h"

#include "data.h"

/* Only for SPATZ_BINARY_START: the binary array itself is defined by the kernel source. */
#define SPATZ_BINARY_NO_DEFINE
#include "onnx_max_task_bin.h"
#undef SPATZ_BINARY_NO_DEFINE

/* Output tensors in L2: every tile writes back its own slice. */
static float16 Y_LAST[BATCH * CHANNELS * HEIGHT]; /* ReduceMax over axis -1 (W) */
static float16 Y_CH[BATCH * HEIGHT * WIDTH];      /* ReduceMax over axis 1 (C)  */

static void sync(fsync_controller_t *fsync_ctrl, eu_controller_t *eu_ctrl)
{
    fsync_sync_global(fsync_ctrl);
    eu_fsync_wait(eu_ctrl, WFE);
}

/* X is viewed as [outer, reduce, inner]: run the kernel on every tile, then check on tile 0. */
static int run_case(const char *name,
                    float16 *Y,
                    const float16 *G,
                    uint32_t outer,
                    uint32_t reduce,
                    uint32_t inner,
                    fsync_controller_t *fsync_ctrl,
                    eu_controller_t *eu_ctrl)
{
    uint32_t cycle_start;
    uint32_t cycle_stop;
    int ret = 0;

    cycle_start = perf_get_cycles();
    MAGIA_reducemax_fp16_spatz(X, Y, outer, reduce, inner);
    cycle_stop = perf_get_cycles();

    printf("[CV32 (%d)] reducemax %s completed in %d cycles\n", HID, name, cycle_stop - cycle_start);

    sync(fsync_ctrl, eu_ctrl);

    if (HID == 0) {
        ret = compare_fp16_bitwise((uintptr_t)Y, (uintptr_t)G, outer * inner);
        printf("[CV32] Case %s %s with %d mismatches\n", name, ret ? "FAILED" : "SUCCESS", ret);
    }

    sync(fsync_ctrl, eu_ctrl);

    return ret;
}

int main(void)
{
    fsync_config_t fsync_cfg      = {.hartid = HID};
    fsync_controller_t fsync_ctrl = {.base = NULL, .cfg = &fsync_cfg, .api = &fsync_api};
    idma_config_t idma_cfg        = {.hartid = HID};
    idma_controller_t idma_ctrl   = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};
    eu_config_t eu_cfg            = {.hartid = HID};
    eu_controller_t eu_ctrl       = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};
    int ret = 0;

    if (HID == 0)
        printf("############################### ONNX_MAX TEST on Tiles "
               "################################\n");

    fsync_init(&fsync_ctrl);
    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    idma_init(&idma_ctrl);
    spatz_init(SPATZ_BINARY_START);

    sync(&fsync_ctrl, &eu_ctrl);

    /* axis -1: contiguous rows (inner = 1), the softmax row-max case */
    ret += run_case(
        "axis=-1", Y_LAST, G_LAST, BATCH * CHANNELS * HEIGHT, WIDTH, 1, &fsync_ctrl, &eu_ctrl);

    /* axis 1: strided reduction over channels (inner = H*W) */
    ret += run_case(
        "axis=1", Y_CH, G_CH, BATCH, CHANNELS, HEIGHT * WIDTH, &fsync_ctrl, &eu_ctrl);

    spatz_clk_dis();

    if (HID == 0)
        printf("[CV32] Test %s\n", ret ? "FAILED" : "SUCCESS");

    return ret;
}
