#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <ctime>
#include <cstring>

using namespace std;
using namespace cv;

#define SIZE 256

// Function to Compute Histogram
void computeHist(const Mat& img, int* hist) {
    memset(hist, 0, SIZE * sizeof(int)); // Initialize histogram bins to 0
    for (int i = 0; i < img.total(); i++) {
        hist[img.data[i]]++;
    }
}

// Function to Compute CDF with Proper Normalization
void computeCDF(int* hist, int* cdf, int total) {
    int sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += hist[i];
        cdf[i] = min(255, max(0, (sum * 255) / total));
    }
}

// Function to Apply Histogram Equalization
void equalize(const Mat& img, Mat& result, int* cdf) {
    for (int i = 0; i < img.total(); i++) {
        result.data[i] = cdf[img.data[i]];
    }
}

int main() {
    // Load Image in Grayscale
    Mat img = imread("Sandesh.jpeg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    // Resize Image to 1024x1024
    resize(img, img, Size(1024, 1024));
    int w = img.cols, h = img.rows, total = w * h;
    int hist[SIZE] = {0}, cdf[SIZE] = {0};
    Mat result(h, w, CV_8U);

    // Start Timing
    clock_t start = clock();

    // Perform Histogram Equalization
    computeHist(img, hist);
    computeCDF(hist, cdf, total);
    equalize(img, result, cdf);

    // End Timing
    clock_t end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC * 1000;

    cout << "\n<<<<<< ..... Histogram Equalization (CPU) ..... >>>>>>\n";
    cout << "\nTime taken: " << duration << " ms" << endl;

    // Calculate Old & New Image Pixel Intensities
    map<int, int> oldIntensities, newIntensities;
    for (int i = 0; i < total; i++) oldIntensities[img.data[i]]++;
    for (int i = 0; i < total; i++) newIntensities[result.data[i]]++;

    // Compute Average Pixels Per Intensity
    int old_total_used_intensities = oldIntensities.size();
    int new_total_used_intensities = newIntensities.size();

    double old_avg_pixels_per_intensity = old_total_used_intensities > 0 ? 
        (double)total / old_total_used_intensities : 0;
    double new_avg_pixels_per_intensity = new_total_used_intensities > 0 ? 
        (double)total / new_total_used_intensities : 0;

    // Display Old Image Pixel Intensities
    cout << "\n\n<<<<<< ..... Old Image Pixel Intensities ..... >>>>>>\n" << endl;
    cout << "Average Pixels Per Intensity: " << old_avg_pixels_per_intensity << " pixels\n" << endl;
    for (auto& pair : oldIntensities) {
        cout << "Intensity " << pair.first << " : " << pair.second << " pixels" << endl;
    }

    // Display New Image Pixel Intensities
    cout << "\n\n<<<<<< ..... New Image Pixel Intensities ..... >>>>>>\n" << endl;
    cout << "Average Pixels Per Intensity: " << new_avg_pixels_per_intensity << " pixels\n" << endl;
    for (auto& pair : newIntensities) {
        cout << "Intensity " << pair.first << " : " << pair.second << " pixels" << endl;
    }

    // Save Equalized Image
    imwrite("result_cpu.jpg", result);

    return 0;
}
