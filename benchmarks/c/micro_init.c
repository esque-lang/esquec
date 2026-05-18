// Micro-benchmark: Tensor literal initialization comparison
// Compile: gcc -O3 -mavx2 -S -o micro_init.s micro_init.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define ITERATIONS 100000000

// Scalar init (what Esque generates)
__attribute__((noinline))
void init_scalar(float* out) {
    out[0] = 1.0f;
    out[1] = 2.0f;
    out[2] = 3.0f;
    out[3] = 4.0f;
    out[4] = 5.0f;
    out[5] = 6.0f;
    out[6] = 7.0f;
    out[7] = 8.0f;
}

// Optimal init (what Esque should generate)
static const float __attribute__((aligned(32))) const_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

__attribute__((noinline))
void init_simd(float* out) {
    __m256 v = _mm256_load_ps(const_data);
    _mm256_storeu_ps(out, v);
}

int main() {
    float __attribute__((aligned(32))) a[8];

    struct timespec start, end;

    // Scalar version
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        init_scalar(a);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double scalar_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Scalar init: %.3f ms (a[0]=%.1f)\n", scalar_time * 1000, a[0]);

    // SIMD version
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        init_simd(a);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double simd_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("SIMD init:   %.3f ms (a[0]=%.1f)\n", simd_time * 1000, a[0]);

    printf("\nSIMD speedup: %.2fx\n", scalar_time / simd_time);

    return 0;
}
