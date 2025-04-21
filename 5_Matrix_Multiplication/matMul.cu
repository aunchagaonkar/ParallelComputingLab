#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

void initializeMatrix(int* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 2;
    }
}

void printMatrix(int* matrix, int N) {
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 15; j++) {
            cout << matrix[i * 15 + j] << " ";
        }
        cout << endl;
    }
}

void matrixMultiply(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 1024;

    int* A = new int[N * N];
    int* B = new int[N * N];
    int* C = new int[N * N];

    srand(time(0));
    initializeMatrix(A, N);
    initializeMatrix(B, N);

    cout << "Matrix A:" << endl;
    printMatrix(A, N);
    
    cout << "Matrix B:" << endl;
    printMatrix(B, N);

    clock_t start = clock();
    matrixMultiply(A, B, C, N);
    clock_t end = clock();

    cout << "Result Matrix C (A * B):" << endl;
    printMatrix(C, N);

    double elapsed = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "Execution Time: " << elapsed << " ms" << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
