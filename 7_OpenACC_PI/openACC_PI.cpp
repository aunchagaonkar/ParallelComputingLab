#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <math.h>
#include <ctime>

#define N 1000000

int main() {
    int count = 0;
    
    // Start timing using clock()
    clock_t start_time = clock();

    #pragma acc parallel loop reduction(+:count)
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

    printf("Time taken: %f ms\n", time_taken); // Output in milliseconds
    printf("Estimated value of Pi: %f\n", pi);

    return 0;
}
