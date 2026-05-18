// Micro-benchmark: Reduction comparison
// This shows the difference between scalar and SIMD reduction
// Compile: gcc -O3 -mavx2 -o micro_reduce micro_reduce.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 8
#define ITERATIONS 100000000

// Scalar reduction (what Esque generates)
__attribute__((noinline))
float reduce_scalar(float* a) {
    float sum = 0.0f;
    sum += a[0];
    sum += a[1];
    sum += a[2];
    sum += a[3];
    sum += a[4];
    sum += a[5];
    sum += a[6];
    sum += a[7];
    return sum;
}

// SIMD reduction (what Esque should generate)
__attribute__((noinline))
float reduce_simd(float* a) {
    __m256 v = _mm256_loadu_ps(a);
    // Horizontal sum: 8 floats -> 1 float
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

int main() {
    float __attribute__((aligned(32))) a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    struct timespec start, end;
    volatile float result;

    // Scalar version
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        result = reduce_scalar(a);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double scalar_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Scalar reduction: %.3f ms (result=%.1f)\n", scalar_time * 1000, result);

    // SIMD version
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        result = reduce_simd(a);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double simd_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("SIMD reduction:   %.3f ms (result=%.1f)\n", simd_time * 1000, result);

    printf("\nSIMD speedup: %.2fx\n", scalar_time / simd_time);

    return 0;
}
