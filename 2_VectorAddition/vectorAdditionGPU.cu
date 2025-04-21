// Simplified CUDA Vector Addition using <ctime>

#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

constexpr int VECTOR_SIZE = 1'000'000;
constexpr int THREADS_PER_BLOCK = 256;

__global__ void vectorAdd(const int *A, const int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    size_t bytes = VECTOR_SIZE * sizeof(int);
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int blocks = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    clock_t start = clock();
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, VECTOR_SIZE);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    bool valid = true;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            valid = false;
            break;
        }
    }

    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    std::cout << "CUDA execution time: " << ms << " ms\n";
    std::cout << (valid ? "Result is correct!" : "Result is incorrect.") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
