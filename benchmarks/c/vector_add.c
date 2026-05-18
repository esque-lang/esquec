// Vector addition benchmark - C with AVX2
// Compile: gcc -O3 -mavx2 -o vector_add vector_add.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 1024
#define ITERATIONS 1000000

__attribute__((noinline))
float vector_add_sum(float* a, float* b, float* c, int n) {
    // AVX2 vectorized addition
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    // Scalar cleanup
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    // Sum reduction
    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        sum += c[i];
    }
    return sum;
}

int main() {
    float *a = aligned_alloc(32, N * sizeof(float));
    float *b = aligned_alloc(32, N * sizeof(float));
    float *c = aligned_alloc(32, N * sizeof(float));

    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = (float)(i % 100) * 0.01f;
        b[i] = (float)(i % 50) * 0.02f;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile float result = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        result = vector_add_sum(a, b, c, N);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C AVX2 vector_add: %.3f ms (%d iterations, N=%d)\n",
           elapsed * 1000, ITERATIONS, N);
    printf("Checksum: %.2f\n", result);
    printf("Throughput: %.2f M elements/sec\n",
           (double)ITERATIONS * N / elapsed / 1e6);

    free(a);
    free(b);
    free(c);
    return 0;
}
