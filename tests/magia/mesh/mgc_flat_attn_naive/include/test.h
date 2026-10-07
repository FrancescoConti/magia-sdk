// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Flat-Attention test - placeholder sizes, zero-filled inputs and all-zero golden output

#ifndef _TEST_FLAT_ATTN_INCLUDE_GUARD_
#define _TEST_FLAT_ATTN_INCLUDE_GUARD_

#define SEQ_LENGTH (256)
#define EMBED_SIZE (64)
#define PROJ_SIZE  (64)
#define NUM_HEADS  (1)

uint16_t q_inp [SEQ_LENGTH * PROJ_SIZE] = {0};
uint16_t k_inp [SEQ_LENGTH * PROJ_SIZE] = {0};
uint16_t v_inp [SEQ_LENGTH * PROJ_SIZE] = {0};
uint16_t o_out [SEQ_LENGTH * PROJ_SIZE] = {0};
uint16_t o_gold[SEQ_LENGTH * PROJ_SIZE] = {0};

#endif //_TEST_FLAT_ATTN_INCLUDE_GUARD_
