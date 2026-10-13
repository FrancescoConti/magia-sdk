#ifndef COLSUM_RESCALE_FP16_SPATZ_PARAMS_H_
#define COLSUM_RESCALE_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_S; /* Row 0 of the sum [cols], in-place                 */
    uintptr_t shard_G; /* Matrix [rows x cols], row 0 unused                */
    uintptr_t shard_P; /* Previous sum [cols], in-place, 0 if there is none */
    uintptr_t shard_A; /* Rescaling factors of P [cols]                     */

    uint32_t rows; /* Number of rows */
    uint32_t cols; /* Row length     */
} colsum_rescale_fp16_spatz_params_t;

#endif /* COLSUM_RESCALE_FP16_SPATZ_PARAMS_H_ */
