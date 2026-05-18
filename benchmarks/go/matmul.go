//go:build ignore

// Matrix multiplication benchmark - Go
// Run: go run matmul.go
package main

import (
	"fmt"
	"time"
)

const N = 64
const ITERATIONS = 10000

//go:noinline
func matmulNaive(A, B, C []float32, n int) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for k := 0; k < n; k++ {
				sum += A[i*n+k] * B[k*n+j]
			}
			C[i*n+j] = sum
		}
	}
}

//go:noinline
func matrixSum(C []float32) float32 {
	var sum float32
	for _, v := range C {
		sum += v
	}
	return sum
}

func main() {
	A := make([]float32, N*N)
	B := make([]float32, N*N)
	C := make([]float32, N*N)

	// Initialize
	for i := 0; i < N*N; i++ {
		A[i] = float32(i%10) * 0.1
		B[i] = float32(i%7) * 0.1
	}

	start := time.Now()

	var result float32
	for iter := 0; iter < ITERATIONS; iter++ {
		matmulNaive(A, B, C, N)
		result = matrixSum(C)
	}

	elapsed := time.Since(start)

	fmt.Printf("Go matmul: %.3f ms (%d iterations, N=%d)\n",
		float64(elapsed.Milliseconds()), ITERATIONS, N)
	fmt.Printf("Checksum: %.2f\n", result)
	fmt.Printf("GFLOPS: %.2f\n",
		float64(ITERATIONS)*2*N*N*N/elapsed.Seconds()/1e9)
}
