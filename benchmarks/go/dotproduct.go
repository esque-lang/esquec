//go:build ignore

// Dot product benchmark - Go
// Run: go run dotproduct.go
package main

import (
	"fmt"
	"time"
)

const N = 1024
const ITERATIONS = 1000000

//go:noinline
func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func main() {
	a := make([]float32, N)
	b := make([]float32, N)

	// Initialize
	for i := 0; i < N; i++ {
		a[i] = float32(i%100) * 0.01
		b[i] = float32(i%50) * 0.02
	}

	start := time.Now()

	var result float32
	for iter := 0; iter < ITERATIONS; iter++ {
		result = dotProduct(a, b)
	}

	elapsed := time.Since(start)

	fmt.Printf("Go dotproduct: %.3f ms (%d iterations, N=%d)\n",
		float64(elapsed.Milliseconds()), ITERATIONS, N)
	fmt.Printf("Checksum: %.6f\n", result)
	fmt.Printf("Throughput: %.2f M elements/sec\n",
		float64(ITERATIONS)*N/elapsed.Seconds()/1e6)
}
