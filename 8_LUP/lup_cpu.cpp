#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

void print_matrix(const char *name, float *mat, int size)
{
  printf("\n%s (Top %dx%d block):\n", name, size, size);
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      printf("%8.2f ", mat[i * N + j]);
    }
    printf("\n");
  }
}

int main()
{
  float *A, *L, *U;
  A = (float *)malloc(N * N * sizeof(float));
  L = (float *)calloc(N * N, sizeof(float));
  U = (float *)calloc(N * N, sizeof(float));

  srand(time(NULL));
  for (int i = 0; i < N * N; i++)
  {
    A[i] = (float)(rand() % 1000) / 10.0f;
  }

  for (int i = 0; i < N; i++)
  {
    L[i * N + i] = 1.0f;
  }

  clock_t start = clock();

  for (int i = 0; i < N; i++)
  {
    for (int j = i; j < N; j++)
    {
      float sum = 0.0f;
      for (int k = 0; k < i; k++)
      {
        sum += L[i * N + k] * U[k * N + j];
      }
      U[i * N + j] = A[i * N + j] - sum;
    }

    for (int j = i + 1; j < N; j++)
    {
      float sum = 0.0f;
      for (int k = 0; k < i; k++)
      {
        sum += L[j * N + k] * U[k * N + i];
      }
      L[j * N + i] = (A[j * N + i] - sum) / U[i * N + i];
    }
  }

  clock_t end = clock();
  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("\nCPU LU decomposition time: %.6f seconds\n", time_spent);

  int print_size = 8;
  print_matrix("Original Matrix A", A, print_size);
  print_matrix("Lower Matrix L", L, print_size);
  print_matrix("Upper Matrix U", U, print_size);

  free(A);
  free(L);
  free(U);

  return 0;
}

/*

g++ lup_cpu.cpp -O3 -o cpu_lu && ./cpu_lu


*/