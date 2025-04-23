#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024

// Function to compress the image by downsampling
void compressImage(const cv::Mat& inputImg, cv::Mat& outputImg, int factor) {
    int newSize = WIDTH / factor;
    outputImg.create(newSize, newSize, CV_8UC1);
    
    // Loop over each pixel of the output image
    for (int y = 0; y < newSize; ++y) {
        for (int x = 0; x < newSize; ++x) {
            int sum = 0;
            // Loop over the region of the input image corresponding to the output pixel
            for (int i = 0; i < factor; i++) {
                for (int j = 0; j < factor; j++) {
                    sum += inputImg.at<uchar>((y * factor + j), (x * factor + i));
                }
            }
            // Calculate the average of the region and store it in the output image
            outputImg.at<uchar>(y, x) = sum / (factor * factor);
        }
    }
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
    cv::imwrite("compressed_2x_CPU.jpg", outputImg2);
    cv::imwrite("compressed_4x_CPU.jpg", outputImg4);
    
    // End measuring time
    clock_t end = clock();
    
    // Calculate and print the elapsed time
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Compression complete! Time taken: " << elapsed_time << " seconds" << std::endl;

    return 0;
}
