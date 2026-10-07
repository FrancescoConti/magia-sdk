// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"

#include "kernels_compare_utils.h"
#include "mul_broadcast_fp16_spatz.h"

#include "data.h"

/* Only for SPATZ_BINARY_START: the binary array itself is defined by the kernel source. */
#define SPATZ_BINARY_NO_DEFINE
#include "onnx_mul_broadcast_task_bin.h"
#undef SPATZ_BINARY_NO_DEFINE

#define TENSOR_LEN (BATCH * CHANNELS * HEIGHT * WIDTH)

/* Output tensor in L2: every tile writes back its own slice. */
static float16 C[TENSOR_LEN];

static void sync(fsync_controller_t *fsync_ctrl, eu_controller_t *eu_ctrl)
{
    fsync_sync_global(fsync_ctrl);
    eu_fsync_wait(eu_ctrl, WFE);
}

int main(void)
{
    fsync_config_t fsync_cfg      = {.hartid = HID};
    fsync_controller_t fsync_ctrl = {.base = NULL, .cfg = &fsync_cfg, .api = &fsync_api};
    idma_config_t idma_cfg        = {.hartid = HID};
    idma_controller_t idma_ctrl   = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};
    eu_config_t eu_cfg            = {.hartid = HID};
    eu_controller_t eu_ctrl       = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};
    uint32_t cycle_start;
    uint32_t cycle_stop;
    int ret = 0;

    if (HID == 0)
        printf("########################## ONNX_MUL_BROADCAST TEST on Tiles "
               "##########################\n");

    fsync_init(&fsync_ctrl);
    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    idma_init(&idma_ctrl);
    spatz_init(SPATZ_BINARY_START);

    sync(&fsync_ctrl, &eu_ctrl);

    cycle_start = perf_get_cycles();
    MAGIA_mul_broadcast_fp16_spatz(A, B, C, TENSOR_LEN);
    cycle_stop = perf_get_cycles();

    printf("[CV32 (%d)] mul_broadcast completed in %d cycles\n", HID, cycle_stop - cycle_start);

    sync(&fsync_ctrl, &eu_ctrl);

    spatz_clk_dis();

    if (HID == 0) {
        ret = compare_fp16_bitwise((uintptr_t)C, (uintptr_t)G, TENSOR_LEN);
        printf("[CV32] Test %s with %d mismatches\n", ret ? "FAILED" : "SUCCESS", ret);
    }

    sync(&fsync_ctrl, &eu_ctrl);

    return ret;
}
