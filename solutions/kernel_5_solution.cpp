#include <iostream>
#include <cmath>
#include <cassert>

#include "test_sgemm2.cu"


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_blocktiling_2d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  const uint totalResultsBlocktile = BM * BN;

  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
  assert(numThreadsBlocktile == blockDim.x);

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  const uint strideA = numThreadsBlocktile / BK;
  const uint strideB = numThreadsBlocktile / BN;

  for (uint blockIdx = 0; blockIdx < K; blockIdx += BK) {

    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
    }

    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }

    }
    __syncthreads();

  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }

}

int main() {
    int M = 4096, N = 4096, K = 4096;

    std::cout << "testing SGEMM with 2D Blocktiling: " << std::endl;

    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    test_sgemm(M, N, K, gridDim, blockDim, sgemm_blocktiling_2d<BM, BN, BK, TM, TN>);

    return 0;
}
