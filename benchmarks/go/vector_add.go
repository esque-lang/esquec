//go:build ignore

// Vector addition benchmark - Go
// Run: go run vector_add.go
package main

import (
	"fmt"
	"time"
)

const N = 1024
const ITERATIONS = 1000000

//go:noinline
func vectorAddSum(a, b, c []float32) float32 {
	for i := range a {
		c[i] = a[i] + b[i]
	}
	var sum float32
	for _, v := range c {
		sum += v
	}
	return sum
}

func main() {
	a := make([]float32, N)
	b := make([]float32, N)
	c := make([]float32, N)

	// Initialize
	for i := 0; i < N; i++ {
		a[i] = float32(i%100) * 0.01
		b[i] = float32(i%50) * 0.02
	}

	start := time.Now()

	var result float32
	for iter := 0; iter < ITERATIONS; iter++ {
		result = vectorAddSum(a, b, c)
	}

	elapsed := time.Since(start)

	fmt.Printf("Go vector_add: %.3f ms (%d iterations, N=%d)\n",
		float64(elapsed.Milliseconds()), ITERATIONS, N)
	fmt.Printf("Checksum: %.2f\n", result)
	fmt.Printf("Throughput: %.2f M elements/sec\n",
		float64(ITERATIONS)*N/elapsed.Seconds()/1e6)
}
