// Sum reduction benchmark - C with AVX2
// Compile: gcc -O3 -mavx2 -o reduce_sum reduce_sum.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 4096
#define ITERATIONS 500000

__attribute__((noinline))
float reduce_sum_avx2(float* a, int n) {
    __m256 sum8 = _mm256_setzero_ps();

    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        sum8 = _mm256_add_ps(sum8, va);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum8, 1);
    __m128 lo = _mm256_castps256_ps128(sum8);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float sum = _mm_cvtss_f32(sum4);

    // Scalar cleanup
    for (; i < n; i++) {
        sum += a[i];
    }

    return sum;
}

__attribute__((noinline))
float reduce_sum_scalar(float* a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

int main() {
    float *a = aligned_alloc(32, N * sizeof(float));

    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = (float)(i % 100) * 0.001f;
    }

    struct timespec start, end;

    // AVX2 version
    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile float result = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        result = reduce_sum_avx2(a, N);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C AVX2 reduce_sum: %.3f ms (%d iterations, N=%d)\n",
           elapsed * 1000, ITERATIONS, N);
    printf("Checksum: %.6f\n", result);
    printf("Throughput: %.2f M elements/sec\n",
           (double)ITERATIONS * N / elapsed / 1e6);

    // Scalar version
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        result = reduce_sum_scalar(a, N);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = (end.tv_sec - start.tv_sec) +
              (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C scalar reduce_sum: %.3f ms (%d iterations, N=%d)\n",
           elapsed * 1000, ITERATIONS, N);

    free(a);
    return 0;
}
