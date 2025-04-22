#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N 1024
#define TILE_WIDTH 16

// CUDA Kernel: Matrix multiplication C = A * B
__global__ void matMulKernel(const int* A, const int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    const int size = N * N;
    const int bytes = size * sizeof(int);

    // Allocate host memory
    int* h_A = (int*)malloc(bytes);
    int* h_B = (int*)malloc(bytes);
    int* h_C = (int*)malloc(bytes);

    // Initialize input matrices with random 0/1 values
    srand((unsigned)time(NULL));
    for (int i = 0; i < size; ++i) {
        h_A[i] = rand() % 2;
        h_B[i] = rand() % 2;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Timing using CUDA events
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Record end event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Print execution time
    std::cout << "GPU Execution Time: " << elapsedTime << " ms" << std::endl;

    // Print a small portion of C for verification (15x15)
    std::cout << "Result Matrix C (first 15x15):" << std::endl;
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
