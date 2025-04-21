#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include <ctime>  // Include ctime for time measurement

using namespace std;
using namespace cv;

struct Transform {
    int rangeX, rangeY;
    int domainX, domainY;
    double s, b;
};

vector<double> extractBlock(const Mat &img, int x, int y, int size) {
    vector<double> block;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            block.push_back(img.at<uchar>(y + i, x + j));
    return block;
}

vector<double> downsampleBlock(const Mat &img, int x, int y, int size) {
    int M = size / 2;
    vector<double> block;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    sum += img.at<uchar>(y + i * 2 + dy, x + j * 2 + dx);
            block.push_back(sum / 4.0);
        }
    }
    return block;
}

void computeAffine(const vector<double> &D, const vector<double> &R, double &s, double &b) {
    int n = D.size();
    double meanD = accumulate(D.begin(), D.end(), 0.0) / n;
    double meanR = accumulate(R.begin(), R.end(), 0.0) / n;
    double cov = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) {
        cov += (D[i] - meanD) * (R[i] - meanR);
        var += (D[i] - meanD) * (D[i] - meanD);
    }
    s = (fabs(var) < 1e-6) ? 0 : cov / var;
    b = meanR - s * meanD;
}

void compress(const cv::Mat &img, int rangeSize, std::vector<Transform> &transforms) {
    int domainSize = rangeSize * 2;
    int totalBlocks = 0;

    // Start measuring time for compress
    std::clock_t start = std::clock();

    for (int ry = 0; ry <= img.rows - rangeSize; ry += rangeSize) {
        for (int rx = 0; rx <= img.cols - rangeSize; rx += rangeSize) {
            auto R = extractBlock(img, rx, ry, rangeSize);
            double bestErr = std::numeric_limits<double>::max();
            Transform bestT = {rx, ry, 0, 0, 0, 0};

            for (int dy = 0; dy <= img.rows - domainSize; dy += 4) {  // Speed up with step of 4
                for (int dx = 0; dx <= img.cols - domainSize; dx += 4) {
                    auto D = downsampleBlock(img, dx, dy, domainSize);
                    double s, b;
                    computeAffine(D, R, s, b);
                    double err = 0;
                    for (int i = 0; i < D.size(); ++i)
                        err += pow(s * D[i] + b - R[i], 2);

                    if (err < bestErr) {
                        bestErr = err;
                        bestT.domainX = dx;
                        bestT.domainY = dy;
                        bestT.s = s;
                        bestT.b = b;
                    }
                }
            }
            transforms.push_back(bestT);
            if (++totalBlocks % 100 == 0)
                std::cout << "Processed " << totalBlocks << " range blocks...\n";
        }
    }

    // End measuring time for compress
    std::clock_t end = std::clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Compression time: " << duration << " seconds.\n";
}

void decompress(const vector<Transform> &transforms, int width, int height, int rangeSize, int iterations, Mat &outImg) {
    outImg = Mat::zeros(height, width, CV_8UC1);
    int domainSize = rangeSize * 2;

    // Start measuring time for decompress
    std::clock_t start = std::clock();

    for (int it = 0; it < iterations; ++it) {
        Mat newImg = outImg.clone();
        for (const auto &T : transforms) {
            vector<double> D = downsampleBlock(outImg, T.domainX, T.domainY, domainSize);
            int idx = 0;
            for (int i = 0; i < rangeSize; ++i) {
                for (int j = 0; j < rangeSize; ++j) {
                    double val = T.s * D[idx++] + T.b;
                    val = min(max(val, 0.0), 255.0);
                    newImg.at<uchar>(T.rangeY + i, T.rangeX + j) = static_cast<uchar>(val);
                }
            }
        }
        outImg = newImg.clone();
    }

    // End measuring time for decompress
    std::clock_t end = std::clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Decompression time: " << duration << " seconds.\n";
}

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <input.png/jpeg> <output.png> <iterations>\n";
        return 1;
    }

    string inPath = argv[1];
    string outPath = argv[2];
    int iterations = stoi(argv[3]);

    Mat img = imread(inPath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Failed to load image!\n";
        return 1;
    }

    int rangeSize = 4;
    vector<Transform> transforms;

    compress(img, rangeSize, transforms);

    Mat outImg;
    decompress(transforms, img.cols, img.rows, rangeSize, iterations, outImg);

    imwrite(outPath, outImg);
    cout << "Output written to: " << outPath << endl;

    return 0;
}
