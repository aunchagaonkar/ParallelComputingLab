#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

const int N = 1024;
const int BLOCK_SIZE = 256;

__global__ void compute_u(float* A, float* L, float* U, int i) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + i;
    if (col < N) {
        float sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i * N + j] * U[j * N + col];
        }
        U[i * N + col] = A[i * N + col] - sum;
    }
}

__global__ void compute_l(float* A, float* L, float* U, int i) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    if (row < N) {
        float sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[row * N + j] * U[j * N + i];
        }
        L[row * N + i] = (A[row * N + i] - sum) / U[i * N + i];
    }
}

void print_matrix(const char* name, float* mat, int size) {
    cout << "\n" << name << " (Top " << size << "x" << size << " block):\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.2f ", mat[i * N + j]);
        }
        cout << "\n";
    }
}

int main() {
    float *A, *L, *U;
    float *d_A, *d_L, *d_U;

    A = new float[N * N];
    L = new float[N * N];
    U = new float[N * N];

    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 1000) / 10.0f;
        L[i] = 0.0f;
        U[i] = 0.0f;
    }

    // Set L as identity matrix
    for (int i = 0; i < N; i++) {
        L[i * N + i] = 1.0f;
    }

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_L, N * N * sizeof(float));
    cudaMalloc(&d_U, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < N; i++) {
        int blocks = (N - i + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_u<<<blocks, BLOCK_SIZE>>>(d_A, d_L, d_U, i);
        if (i < N - 1)
            compute_l<<<blocks, BLOCK_SIZE>>>(d_A, d_L, d_U, i);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(L, d_L, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nCUDA LU decomposition time: " << milliseconds / 1000.0f << " seconds\n";

    int print_size = 8;
    print_matrix("Original Matrix A", A, print_size);
    print_matrix("Lower Matrix L", L, print_size);
    print_matrix("Upper Matrix U", U, print_size);

    delete[] A;
    delete[] L;
    delete[] U;

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    return 0;
}


/* 
nvcc lup_cuda.cu -O3 -o cuda_lu && ./cuda_lu 
*/