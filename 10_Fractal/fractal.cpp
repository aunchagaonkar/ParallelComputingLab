#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;  // To avoid writing std:: everywhere
using namespace cv;   // To avoid writing cv:: everywhere

void reduce_image(sycl::queue& q, const vector<float>& input_image, vector<float>& output_image, int width, int height, int factor) {
    int output_width = width / factor, output_height = height / factor;

    sycl::buffer<float> input_buf(input_image.data(), sycl::range<1>(input_image.size()));
    sycl::buffer<float> output_buf(output_image.data(), sycl::range<1>(output_image.size()));

    q.submit([&](sycl::handler& cgh) {
        auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
        auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class reduce_kernel>(sycl::range<2>(output_height, output_width), [=](sycl::item<2> item) {
            int y = item.get_id(0), x = item.get_id(1);
            float sum = 0.0f;

            for (int dy = 0; dy < factor; ++dy)
                for (int dx = 0; dx < factor; ++dx) {
                    int input_x = x * factor + dx, input_y = y * factor + dy;
                    if (input_x < width && input_y < height)
                        sum += input_acc[input_y * width + input_x];
                }
            output_acc[y * output_width + x] = sum / (factor * factor);
        });
    }).wait();
}

int main() {
    // Load image (grayscale)
    Mat img = imread("Sandesh.jpeg", IMREAD_GRAYSCALE);
    if (img.empty()) return -1;

    int width = img.cols, height = img.rows, factor = 4;

    // Flatten image to vector
    vector<float> input_img(width * height);
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            input_img[i * width + j] = img.at<uchar>(i, j);

    // Prepare output image
    vector<float> output_img((width / factor) * (height / factor));

    // Reduce image resolution using SYCL
    sycl::queue q;
    reduce_image(q, input_img, output_img, width, height, factor);

    // Convert output to 8-bit for saving
    Mat output_mat(height / factor, width / factor, CV_32F, output_img.data());
    output_mat.convertTo(output_mat, CV_8U);

    // Save reduced image with compression
    imwrite("output_image.png", output_mat, {IMWRITE_PNG_COMPRESSION, 3});

    return 0;
}
