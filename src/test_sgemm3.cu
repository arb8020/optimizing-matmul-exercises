#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    for (int i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.01) {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n", matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

void test_sgemm(int M, int N, int K, dim3 gridDim, dim3 blockDim, void (*sgemm_kernel)(int, int, int, float, float*, float*, float, float*)) {
    float *A, *B, *C, *C_ref;
    cudaError_t err;

    err = cudaMallocManaged(&A, M * K * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for matrix A (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged(&B, K * N * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for matrix B (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged(&C, M * N * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for matrix C (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err = cudaMallocManaged(&C_ref, M * N * sizeof(float));
    if (err != cudaSuccess) {
        printf("Failed to allocate memory for matrix C_ref (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }


    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));
    cudaMallocManaged(&C_ref, M * N * sizeof(float));




    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; i++) A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) B[i] = dis(gen);
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
        C_ref[i] = 0.0f;
    }
    std::cout << "generated matrices" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaEventRecord(start);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to record start event (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    sgemm_kernel<<<gridDim, blockDim>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err = cudaEventRecord(stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to record stop event (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize (error code %d)!\n", cudaGetLastError());
        exit(EXIT_FAILURE);
    }


    int numIterations = 10;
    float totalTime = 0.0f;

    for (int i = 0; i < numIterations; i++) {
        cudaEventRecord(start);
        sgemm_kernel<<<gridDim, blockDim>>>(M, N, K, 1.0f, A, B, 0.0f, C);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }

    float avgTime = totalTime / numIterations;
    float gflops = (2.0f * M * N * K) / (avgTime * 1e-3f * 1e9f);

    std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "Average execution time: " << avgTime << " ms" << std::endl;
    std::cout << "Average Performance: " << gflops << " GFLOPS" << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C_ref, N);
    cudaDeviceSynchronize();

    bool correct = verify_matrix(C_ref, C, M * N);
    if (correct) {
        std::cout << "The matrix multiplication is correct!" << std::endl;
    } else {
        std::cout << "There is an error in the matrix multiplication!" << std::endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_ref);
    cublasDestroy(handle);
}
