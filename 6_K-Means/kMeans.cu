#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>  // For clock() and CLOCKS_PER_SEC
#include <cmath>
#include <float.h>
#include <cuda_runtime.h>

#define N 150
#define D 4
#define K 3
#define MAX_ITER 100

float *data;
float *centroids;
int *labels;

__global__ void assign_clusters(float *data, float *centroids, int *labels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float min_dist = FLT_MAX;
        int closest = -1;
        for (int k = 0; k < K; k++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = data[idx * D + d] - centroids[k * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                closest = k;
            }
        }
        labels[idx] = closest;
    }
}

__global__ void update_centroids(float *data, float *centroids, int *labels) {
    __shared__ float centroid_sums[K][D];
    __shared__ int counts[K];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) {
        for (int k = 0; k < K; k++) {
            counts[k] = 0;
            for (int d = 0; d < D; d++) {
                centroid_sums[k][d] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (tid < N) {
        int label = labels[tid];
        for (int d = 0; d < D; d++) {
            atomicAdd(&centroid_sums[label][d], data[tid * D + d]);
        }
        atomicAdd(&counts[label], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                for (int d = 0; d < D; d++) {
                    centroids[k * D + d] = centroid_sums[k][d] / counts[k];
                }
            }
        }
    }
}

void load_data(const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    data = new float[N * D];
    centroids = new float[K * D];
    labels = new int[N];

    int idx = 0;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }

        std::stringstream ss(line);
        std::string val;
        for (int d = 0; d < D; ++d) {
            std::getline(ss, val, ',');
            data[idx * D + d] = std::stof(val);
        }
        idx++;
        if (idx >= N) break;
    }

    srand(time(0));
    for (int k = 0; k < K; ++k) {
        int rand_idx = rand() % N;
        for (int d = 0; d < D; ++d) {
            centroids[k * D + d] = data[rand_idx * D + d];
        }
    }
}

int main() {
    float *d_data, *d_centroids;
    int *d_labels;

    load_data("IRIS.csv");

    cudaMalloc(&d_data, N * D * sizeof(float));
    cudaMalloc(&d_centroids, K * D * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));

    cudaMemcpy(d_data, data, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing using clock()
    std::clock_t start = std::clock();

    for (int i = 0; i < MAX_ITER; i++) {
        assign_clusters<<<1, 256>>>(d_data, d_centroids, d_labels);
        update_centroids<<<1, 256>>>(d_data, d_centroids, d_labels);
        cudaDeviceSynchronize();
    }

    // Stop timing using clock()
    std::clock_t end = std::clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000;  // Convert to milliseconds

    cudaMemcpy(centroids, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final centroids:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ": ";
        for (int d = 0; d < D; ++d) {
            std::cout << centroids[k * D + d] << (d < D - 1 ? ", " : "");
        }
        std::cout << "\n";
    }

    std::cout << "Execution time: " << elapsed_time << " ms\n";

    delete[] data;
    delete[] centroids;
    delete[] labels;
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);

    return 0;
}
