#ifndef EXPMINUS_ROWSUM_FP16_SPATZ_H_
#define EXPMINUS_ROWSUM_FP16_SPATZ_H_

#include <stdint.h>

/* Y[r, c] = exp(s * (X[r, c] - M[r])) and S[r] = sum_c Y[r, c]; if A is not NULL, also
   A[r] = exp(s * (A[r] - M[r])) in place. All operands already in this tile's L1. */
void MAGIA_expminus_rowsum_fp16_spatz_l1(const float16 *X,
                                         const float16 *M,
                                         float16 *Y,
                                         float16 *S,
                                         float16 *A,
                                         float16 s,
                                         uint32_t rows,
                                         uint32_t cols);

#endif /* EXPMINUS_ROWSUM_FP16_SPATZ_H_ */
