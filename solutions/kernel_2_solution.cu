#include <iostream>
#include <cmath>
#include "test_sgemm.cu"

// add BLOCKSIZE
template <const uint BLOCKSIZE>
__global__ void sgemm_coalesce(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {



    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
      float tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[cRow * K + i] * B[i * N + cCol];
      }
      C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }

}

int main() {
    int M = 512, N = 512, K = 512;

    std::cout << "testing SGEMM with Coalescing: " << std::endl;

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    test_sgemm(M, N, K, gridDim, blockDim, sgemm_coalesce<32>);

    return 0;
}
