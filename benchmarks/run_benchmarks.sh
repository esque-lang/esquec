#!/bin/bash
# Esque vs C Benchmark Suite

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "Esque vs C Benchmark Suite"
echo "========================================"
echo ""

# Build Esque benchmarks
echo "Building Esque benchmarks..."
mkdir -p /tmp/esque_bench
go run ./cmd/esquec build -o /tmp/esque_bench/vector_add benchmarks/esque/bench_vector_add.esq

# Build C benchmarks
echo "Building C benchmarks..."
mkdir -p /tmp/c_bench
gcc -O3 -mavx2 -o /tmp/c_bench/vector_add benchmarks/c/bench_vector_add.c

echo ""
echo "========================================"
echo "Benchmark: 8-element Vector Addition (10000 iterations)"
echo "========================================"

# Time C version (multiple runs for accuracy)
echo ""
echo "C (gcc -O3 -mavx2):"
for i in 1 2 3; do
    /tmp/c_bench/vector_add 2>&1 | grep "vector_add" || true
done

# Time Esque version
echo ""
echo "Esque:"
for i in 1 2 3; do
    start=$(date +%s.%N)
    /tmp/esque_bench/vector_add || true
    end=$(date +%s.%N)
    elapsed=$(awk "BEGIN {printf \"%.6f\", $end - $start}")
    echo "  Run $i: ${elapsed}s"
done

echo ""
echo "========================================"
echo "Generated Assembly Analysis"
echo "========================================"
echo ""
echo "Esque AVX2 instructions in generated binary:"
objdump -d /tmp/esque_bench/vector_add 2>/dev/null | grep -E "vaddps|vmovups|vsubps|vmulps" | head -20
