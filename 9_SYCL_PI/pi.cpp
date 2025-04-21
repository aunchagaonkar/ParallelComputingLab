#include <iostream>

#include <cmath>
#include <ctime>

#define N 1000000

int main() {
    int count = 0;
    
    // Start timing using clock()
    clock_t start_time = clock();

  
    for (int i = 0; i < N; i++) {
        double x = ((double) rand() / RAND_MAX);
        double y = ((double) rand() / RAND_MAX);
        double inCircle = x * x + y * y;
        
        if (inCircle <= 1.0) {
            count++;
        }
    }

    double pi = 4.0 * count / N;
    
    // End timing using clock()
    clock_t end_time = clock();
    
    double time_taken = double(end_time - start_time) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    std::cout << "Time taken: " << time_taken << " ms" << std::endl; // Output in milliseconds
    std::cout << "Estimated value of Pi: " << pi << std::endl;

    return 0;
}
