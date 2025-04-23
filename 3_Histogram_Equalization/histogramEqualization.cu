#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <ctime> // Include ctime for clock()

using namespace std;

#define SIZE 256

// **CUDA Kernel: Compute Histogram**
__global__ void computeHist(unsigned char* img, int* hist, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) 
        atomicAdd(&hist[img[idx]], 1);
}

// **CUDA Kernel: Compute CDF**
__global__ void computeCDF(int* hist, int* cdf, int total) {
    int sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += hist[i]; // Accumulate histogram values
        cdf[i] = (sum * 255) / total; // Normalize CDF to [0, 255]
    }
}

// **CUDA Kernel: Apply Histogram Equalization**
__global__ void equalize(unsigned char* in, unsigned char* out, int* cdf, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) 
        out[idx] = cdf[in[idx]];
}

// **Main Function**
int main() {
    // **Load Grayscale Image**
    cv::Mat img = cv::imread("Ameya.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    // **Resize Image to 1024x1024**
    cv::resize(img, img, cv::Size(1024, 1024));
    int w = img.cols, h = img.rows, total = w * h;
    uchar *d_in, *d_out;
    int *d_hist, *d_cdf;

    // **Allocate Memory on GPU**
    cudaMalloc(&d_in, total);
    cudaMalloc(&d_out, total);
    cudaMalloc(&d_hist, SIZE * sizeof(int));
    cudaMalloc(&d_cdf, SIZE * sizeof(int));
    cudaMemcpy(d_in, img.data, total, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, SIZE * sizeof(int));

    // **Start Timing (using clock())**
    clock_t start = clock();

    // **Launch CUDA Kernels**
    computeHist<<<(total + 255) / 256, 256>>>(d_in, d_hist, total);
    computeCDF<<<1, 1>>>(d_hist, d_cdf, total);
    equalize<<<(total + 255) / 256, 256>>>(d_in, d_out, d_cdf, total);

    // **Synchronize to Ensure Kernel Execution Completes**
    cudaDeviceSynchronize();

    // **End Timing (using clock())**
    clock_t end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    cout << "\n<<<<<< ..... Histogram Equalization (GPU) ..... >>>>>>\n";
    cout << "\nTime taken: " << duration << " ms" << endl;

    // **Copy the Equalized Image Back to Host**
    cv::Mat result(h, w, CV_8U);
    cudaMemcpy(result.data, d_out, total, cudaMemcpyDeviceToHost);

    // **Calculate Old & New Image Pixel Intensities**
    map<int, int> oldIntensities, newIntensities;
    for (int i = 0; i < total; i++) 
        oldIntensities[img.data[i]]++;
    for (int i = 0; i < total; i++) 
        newIntensities[result.data[i]]++;

    // **Compute Average Pixels Per Intensity**
    int old_total_used_intensities = oldIntensities.size();
    int new_total_used_intensities = newIntensities.size();
    double old_avg_pixels_per_intensity = old_total_used_intensities > 0 ? (double)total / old_total_used_intensities : 0;
    double new_avg_pixels_per_intensity = new_total_used_intensities > 0 ? (double)total / new_total_used_intensities : 0;

    // **Display Old Image Pixel Intensities**
    cout << "\n\n<<<<<< ..... Old Image Pixel Intensities ..... >>>>>>\n" << endl;
    cout << "Average Pixels Per Intensity: " << old_avg_pixels_per_intensity << " pixels\n" << endl;
    for (auto& pair : oldIntensities) {
        cout << "Intensity " << pair.first << " : " << pair.second << " pixels" << endl;
    }

    // **Display New Image Pixel Intensities**
    cout << "\n\n<<<<<< ..... New Image Pixel Intensities ..... >>>>>>\n" << endl;
    cout << "Average Pixels Per Intensity: " << new_avg_pixels_per_intensity << " pixels\n" << endl;
    for (auto& pair : newIntensities) {
        cout << "Intensity " << pair.first << " : " << pair.second << " pixels" << endl;
    }

    // **Save the Equalized Image**
    cv::imwrite("result_GPU.jpg", result);

    // **Free GPU Memory**
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}
