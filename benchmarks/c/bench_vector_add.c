// Benchmark: 8-element vector addition, 10000 iterations
// Equivalent to Esque bench_vector_add.esq
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#define ITERATIONS 10000

int main() {
    float __attribute__((aligned(32))) a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float __attribute__((aligned(32))) b[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // AVX2 version
    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);

    for (int i = 0; i < ITERATIONS; i++) {
        va = _mm256_add_ps(va, vb);
    }

    _mm256_store_ps(a, va);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Compute checksum
    float sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += a[i];
    }

    printf("C AVX2 vector_add (8 elem, %d iters): %.3f ms\n", ITERATIONS, elapsed * 1000);
    printf("Checksum: %.1f (result: %d)\n", sum, (int)(sum / 80.0f));

    return (int)(sum / 80.0f);
}
