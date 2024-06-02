#ifndef MATRIX_ADD_CUDA_H
#define MATRIX_ADD_CUDA_H

#include "matrix_utils_cuda.h"

void matrix_add_cuda(const float *A, const float *B, const float *C, float *D, int n);

#endif // MATRIX_ADD_CUDA_H