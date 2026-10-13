#ifndef EXPMINUS_ROWSUM_FP16_SPATZ_PARAMS_H_
#define EXPMINUS_ROWSUM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Input matrix [rows x cols]                       */
    uintptr_t shard_M; /* Row max [rows]                                   */
    uintptr_t shard_Y; /* Output matrix [rows x cols], may alias X         */
    uintptr_t shard_S; /* Output row sum of Y [rows]                       */
    uintptr_t shard_A; /* Previous row max [rows], in-place, 0 if none     */

    uint32_t rows;  /* Number of rows                 */
    uint32_t cols;  /* Row length                     */
    uint32_t scale; /* fp16 bits of the scale factor  */
} expminus_rowsum_fp16_spatz_params_t;

#endif /* EXPMINUS_ROWSUM_FP16_SPATZ_PARAMS_H_ */
