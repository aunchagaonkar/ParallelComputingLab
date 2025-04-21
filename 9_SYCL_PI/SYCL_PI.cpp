#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <chrono> // Include the chrono library for timing

using namespace sycl;
using namespace std::chrono;

int main() {
    constexpr size_t num_points = 10'000'000;
    int inside_circle = 0;

    // Record start time
    auto start = high_resolution_clock::now();

    queue q{ default_selector_v };

    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

    buffer<int> count_buf(&inside_circle, 1);

    q.submit([&](handler& h) {
        auto count = count_buf.get_access<access::mode::atomic>(h);

        h.parallel_for(range<1>{num_points}, [=](id<1> i) {
            size_t id = i[0];
            float x = std::fmod(std::abs(std::sin(id * 12.9898f) * 43758.5453f), 1.0f) * 2.0f - 1.0f;
            float y = std::fmod(std::abs(std::cos(id * 78.233f) * 12345.6789f), 1.0f) * 2.0f - 1.0f;

            if (x * x + y * y <= 1.0f)
                count[0].fetch_add(1);
        });
    });

    q.wait();

    float pi = 4.0f * inside_circle / num_points;
    std::cout << "Estimated Pi = " << pi << "\n";

    // Record end time
    auto end = high_resolution_clock::now();

    // Calculate and print elapsed time
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " ms\n";

    return 0;
}
