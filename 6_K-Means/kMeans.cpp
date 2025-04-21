#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <float.h>
#include <ctime>  

#define N 150
#define D 4
#define K 3
#define MAX_ITER 100

float *data;
float *centroids;
int *labels;

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

void assign_clusters(float *data, float *centroids, int *labels) {
    #pragma acc parallel loop collapse(1) present(data, centroids, labels)
    for (int i = 0; i < N; i++) {
        float min_dist = FLT_MAX;
        int closest = -1;
        for (int k = 0; k < K; k++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = data[i * D + d] - centroids[k * D + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                closest = k;
            }
        }
        labels[i] = closest;
    }
}

void update_centroids(float *data, float *centroids, int *labels) {
    float new_centroids[K * D] = {0};
    int counts[K] = {0};

    #pragma acc parallel loop present(data, labels) reduction(+:new_centroids[:K*D], counts[:K])
    for (int i = 0; i < N; i++) {
        int label = labels[i];
        for (int d = 0; d < D; d++) {
            new_centroids[label * D + d] += data[i * D + d];
        }
        counts[label]++;
    }

    #pragma acc parallel loop present(centroids)
    for (int k = 0; k < K; k++) {
        if (counts[k] > 0) {
            for (int d = 0; d < D; d++) {
                centroids[k * D + d] = new_centroids[k * D + d] / counts[k];
            }
        }
    }
}

int main() {
    load_data("IRIS.csv");

    std::clock_t start = std::clock(); // Start timing

    #pragma acc data copyin(data[0:N*D], centroids[0:K*D]) copyout(labels[0:N])
    {
        for (int iter = 0; iter < MAX_ITER; iter++) {
            assign_clusters(data, centroids, labels);
            update_centroids(data, centroids, labels);
        }
    }

    std::clock_t end = std::clock(); // End timing
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

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

    return 0;
}
