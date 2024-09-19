#include <iostream>
#include <cmath>
#include "test_sgemm.cu"

// CUDA kernel function for performing a naive implementation of GEMM (sgemm)
// SOLUTION
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // calculate the x-coordinate of the current thread within the grid
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;

    // calculate the y-coordinate of the current thread within the grid
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // check if the current thread's coordinates are within the bounds of the output matrix C
    if (x < M && y < N) {
        // initialize a temporary variable to store the dot product
        float tmp = 0.0;

        // compute the dot product of the x-th row of matrix A and the y-th column of matrix B
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }

        // compute the final value for the element at position (x, y) in the output matrix C
        // by multiplying the temporary result 'tmp' by 'alpha', adding the product of 'beta'
        // and the existing value at that position in C, and storing the result in C

        // alpha and beta allow us to easily linearly transform the matmul
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    int M = 512, N = 512, K = 512;

    std::cout << "testing naive SGEMM: " << std::endl;

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    test_sgemm(M, N, K, gridDim, blockDim, sgemm_naive);

    return 0;
}
