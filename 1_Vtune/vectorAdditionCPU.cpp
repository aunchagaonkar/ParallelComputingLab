// Simple CPU Vector Addition using <ctime>

#include <iostream>
#include <ctime>

constexpr int VECTOR_SIZE = 1'000'000;

int main() {
    int* A = new int[VECTOR_SIZE];
    int* B = new int[VECTOR_SIZE];
    int* C = new int[VECTOR_SIZE];

    // Initialize input vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        A[i] = 1;
        B[i] = 2;
    }

    // Measure execution time using clock()
    clock_t start = clock();

    // Perform vector addition on CPU
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        C[i] = A[i] + B[i];
    }

    clock_t end = clock();

    // Validate result
    bool valid = true;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        if (C[i] != A[i] + B[i]) {
            valid = false;
            break;
        }
    }

    double ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    std::cout << "CPU execution time: " << ms << " ms\n";
    std::cout << (valid ? "Result is correct!" : "Result is incorrect.") << std::endl;

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
