#include <iostream>
#include <cmath>
#include "test_sgemm.cu"


template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;


  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (int blockIdx = 0; blockIdx < K; blockIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];

      for (uint resultIdx = 0; resultIdx < TM; ++resultIdx) {
        threadResults[resultIdx] += As[(threadRow * TM + resultIdx) * BK + dotIdx] * tmpB;
      }
    }

    A += BK;
    B += BK * N;

    __syncthreads();

  }

  for (uint resultIdx = 0; resultIdx < TM; ++resultIdx) {
    C[(threadRow * TM + resultIdx) * N + threadCol] =
        alpha * threadResults[resultIdx] +
        beta * C[(threadRow * TM + resultIdx) * N + threadCol];
  }


}

int main() {
    int M = 4096, N = 4096, K = 4096;

    std::cout << "testing SGEMM with 1D Blocktiling: " << std::endl;

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / TM);

    test_sgemm(M, N, K, gridDim, blockDim, sgemm_blocktiling_1d<BM, BN, BK, TM>);

    return 0;
}
