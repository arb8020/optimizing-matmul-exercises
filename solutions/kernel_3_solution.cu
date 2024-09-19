#include <iostream>
#include <cmath>
#include "test_sgemm.cu"

template <const uint BLOCKSIZE>
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {


  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  float tmp = 0.0;

  for (int blockIdx = 0; blockIdx < K; blockIdx += BLOCKSIZE) {

    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    __syncthreads();

  }

  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];

}

int main() {
    int M = 512, N = 512, K = 512;

    std::cout << "testing SGEMM with Shared Memory: " << std::endl;

    // make blockDim one dimensional
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    test_sgemm(M, N, K, gridDim, blockDim, sgemm_shared<32>);

    return 0;
}
