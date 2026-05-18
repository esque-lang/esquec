// Matrix multiplication benchmark - C
// Compile: gcc -O3 -mavx2 -o matmul matmul.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 64
#define ITERATIONS 10000

__attribute__((noinline))
void matmul_naive(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__attribute__((noinline))
float matrix_sum(float* C, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n * n; i++) {
        sum += C[i];
    }
    return sum;
}

int main() {
    float *A = aligned_alloc(32, N * N * sizeof(float));
    float *B = aligned_alloc(32, N * N * sizeof(float));
    float *C = aligned_alloc(32, N * N * sizeof(float));

    // Initialize
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 10) * 0.1f;
        B[i] = (float)(i % 7) * 0.1f;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile float result = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        matmul_naive(A, B, C, N);
        result = matrix_sum(C, N);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C matmul (naive): %.3f ms (%d iterations, N=%d)\n",
           elapsed * 1000, ITERATIONS, N);
    printf("Checksum: %.2f\n", result);
    printf("GFLOPS: %.2f\n",
           (double)ITERATIONS * 2 * N * N * N / elapsed / 1e9);

    free(A);
    free(B);
    free(C);
    return 0;
}
