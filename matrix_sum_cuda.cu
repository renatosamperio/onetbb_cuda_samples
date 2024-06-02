#include "matrix_utils_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrix_sum_kernel(const float* A, const float* B, const float* C, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("  Executing kernel on (%d, %d)", idx, idy);
    if (idx < N && idy < N) {
        int index = idy * N + idx;
        result[index] = A[index] + B[index] + C[index];
    }
}

__global__ void matrix_addition(const float* A, const float* B, float* C, float* result, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        result[index] = A[index] + B[index] + C[index];
    }
}


extern "C"
void matrix_sum_cuda(const float* A, const float* B, const float* C, float* result, int N, int idx) {
    float *d_A, *d_B, *d_C, *d_result;
    size_t size = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_result, size));

    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice));

    // dim3 threadsPerBlock(65536, 65536);
    // dim3 threadsPerBlock(1024, 1024);
    // dim3 threadsPerBlock(16, 16);
    dim3 threadsPerBlock(16, 4);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // std::cout << "    CUDA summing in: " << idx << std::endl;
    matrix_sum_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_result, N);
    CUDA_CHECK(cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_result));
}

extern "C"
void gpu_matrix_addition(const float* A, const float* B, const float* C, float* result, int N) {
    float *d_A, *d_B, *d_C, *d_result;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_result, N);

    cudaMemcpy(result, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}