#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024

// CUDA Kernel to downsample the image
__global__ void downsampleKernel(const uchar* input, uchar* output, int w, int factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < w / factor && y < w / factor) {
        int sum = 0;
        for (int i = 0; i < factor; i++)
            for (int j = 0; j < factor; j++)
                sum += input[(y * factor + j) * w + (x * factor + i)];
        
        output[y * (w / factor) + x] = sum / (factor * factor);
    }
}

// Function to compress the image by downsampling
void compressImage(const cv::Mat& inputImg, cv::Mat& outputImg, int factor) {
    int newSize = WIDTH / factor;
    outputImg.create(newSize, newSize, CV_8UC1);
    
    uchar *d_input, *d_output;
    cudaMalloc(&d_input, WIDTH * HEIGHT);
    cudaMalloc(&d_output, newSize * newSize);
    cudaMemcpy(d_input, inputImg.data, WIDTH * HEIGHT, cudaMemcpyHostToDevice);
    
    // Define grid and block size
    dim3 blockSize(16, 16);
    dim3 gridSize((newSize + 15) / 16, (newSize + 15) / 16);
    
    // Launch the kernel
    downsampleKernel<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, factor);
    cudaMemcpy(outputImg.data, d_output, newSize * newSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Load input image
    cv::Mat inputImg = cv::imread("Ameya.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImg.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Resize input image to fixed width and height
    cv::resize(inputImg, inputImg, cv::Size(WIDTH, HEIGHT));

    // Start measuring time
    clock_t start = clock();

    // Compress the image with different factors
    cv::Mat outputImg2, outputImg4;
    compressImage(inputImg, outputImg2, 2);
    compressImage(inputImg, outputImg4, 4);
    
    // Save the compressed images
    cv::imwrite("compressed_2x_GPU.jpg", outputImg2);
    cv::imwrite("compressed_4x_GPU.jpg", outputImg4);
    
    // End measuring time
    clock_t end = clock();
    
    // Calculate and print the elapsed time
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Compression complete! Time taken: " << elapsed_time << " seconds" << std::endl;

    return 0;
}
