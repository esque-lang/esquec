// Package benchmarks provides performance comparison tests for esquec.
package benchmarks

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// BenchmarkResult stores timing information for a benchmark.
type BenchmarkResult struct {
	Name      string
	Language  string
	Duration  time.Duration
	Iterations int
	Elements  int
}

func TestBenchmarkSuite(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("benchmarks require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()

	// Build esquec
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building esquec: %v\n%s", err, out)
	}

	results := make(map[string][]BenchmarkResult)

	// Run vector addition benchmarks
	t.Run("VectorAdd", func(t *testing.T) {
		// C benchmark
		cResult := runCBenchmark(t, tmp, "c/vector_add.c", "vector_add")
		if cResult != nil {
			results["vector_add"] = append(results["vector_add"], *cResult)
		}

		// Go benchmark
		goResult := runGoBenchmark(t, "go/vector_add.go")
		if goResult != nil {
			results["vector_add"] = append(results["vector_add"], *goResult)
		}

		// Esque benchmark - using AVX2 8-element test
		esqResult := runEsqueBenchmark(t, esquec, tmp, "../tests/e2e/avx2_vectorize.esq", "avx2_vectorize", 100000)
		if esqResult != nil {
			results["vector_add"] = append(results["vector_add"], *esqResult)
		}
	})

	// Run dot product benchmarks
	t.Run("DotProduct", func(t *testing.T) {
		// C benchmark
		cResult := runCBenchmark(t, tmp, "c/dotproduct.c", "dotproduct")
		if cResult != nil {
			results["dotproduct"] = append(results["dotproduct"], *cResult)
		}

		// Go benchmark
		goResult := runGoBenchmark(t, "go/dotproduct.go")
		if goResult != nil {
			results["dotproduct"] = append(results["dotproduct"], *goResult)
		}

		// Esque benchmark
		esqResult := runEsqueBenchmark(t, esquec, tmp, "../tests/e2e/dotproduct.esq", "dotproduct", 100000)
		if esqResult != nil {
			results["dotproduct"] = append(results["dotproduct"], *esqResult)
		}
	})

	// Run matrix multiplication benchmarks
	t.Run("MatMul", func(t *testing.T) {
		// C benchmark
		cResult := runCBenchmark(t, tmp, "c/matmul.c", "matmul")
		if cResult != nil {
			results["matmul"] = append(results["matmul"], *cResult)
		}

		// Go benchmark
		goResult := runGoBenchmark(t, "go/matmul.go")
		if goResult != nil {
			results["matmul"] = append(results["matmul"], *goResult)
		}

		// Esque benchmark
		esqResult := runEsqueBenchmark(t, esquec, tmp, "../tests/e2e/matmul.esq", "matmul", 100000)
		if esqResult != nil {
			results["matmul"] = append(results["matmul"], *esqResult)
		}
	})

	// Run reduction benchmarks
	t.Run("ReduceSum", func(t *testing.T) {
		// C benchmark
		cResult := runCBenchmark(t, tmp, "c/reduce_sum.c", "reduce_sum")
		if cResult != nil {
			results["reduce_sum"] = append(results["reduce_sum"], *cResult)
		}

		// Esque benchmark
		esqResult := runEsqueBenchmark(t, esquec, tmp, "../tests/e2e/reduce_sum.esq", "reduce_sum", 100000)
		if esqResult != nil {
			results["reduce_sum"] = append(results["reduce_sum"], *esqResult)
		}
	})

	// Print summary
	printResults(t, results)
}

func runCBenchmark(t *testing.T, tmp, src, name string) *BenchmarkResult {
	t.Helper()

	srcPath, err := filepath.Abs(src)
	if err != nil {
		t.Logf("C benchmark %s: %v", name, err)
		return nil
	}

	if _, err := os.Stat(srcPath); err != nil {
		t.Logf("C source not found: %s", srcPath)
		return nil
	}

	exe := filepath.Join(tmp, name+"_c")
	cmd := exec.Command("gcc", "-O3", "-mavx2", "-march=native", "-o", exe, srcPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("gcc compile failed: %v\n%s", err, out)
		return nil
	}

	var stdout bytes.Buffer
	cmd = exec.Command(exe)
	cmd.Stdout = &stdout
	start := time.Now()
	if err := cmd.Run(); err != nil {
		t.Logf("C benchmark failed: %v", err)
		return nil
	}
	elapsed := time.Since(start)

	t.Logf("C %s output:\n%s", name, stdout.String())

	return &BenchmarkResult{
		Name:     name,
		Language: "C (gcc -O3 -mavx2)",
		Duration: elapsed,
	}
}

func runGoBenchmark(t *testing.T, src string) *BenchmarkResult {
	t.Helper()

	srcPath, err := filepath.Abs(src)
	if err != nil {
		t.Logf("Go benchmark: %v", err)
		return nil
	}

	if _, err := os.Stat(srcPath); err != nil {
		t.Logf("Go source not found: %s", srcPath)
		return nil
	}

	var stdout bytes.Buffer
	cmd := exec.Command("go", "run", srcPath)
	cmd.Stdout = &stdout
	start := time.Now()
	if err := cmd.Run(); err != nil {
		t.Logf("Go benchmark failed: %v", err)
		return nil
	}
	elapsed := time.Since(start)

	name := strings.TrimSuffix(filepath.Base(src), ".go")
	t.Logf("Go %s output:\n%s", name, stdout.String())

	return &BenchmarkResult{
		Name:     name,
		Language: "Go",
		Duration: elapsed,
	}
}

func runEsqueBenchmark(t *testing.T, esquec, tmp, src, name string, iterations int) *BenchmarkResult {
	t.Helper()

	srcPath, err := filepath.Abs(src)
	if err != nil {
		t.Logf("Esque benchmark %s: %v", name, err)
		return nil
	}

	if _, err := os.Stat(srcPath); err != nil {
		t.Logf("Esque source not found: %s", srcPath)
		return nil
	}

	exe := filepath.Join(tmp, name+"_esq")
	cmd := exec.Command(esquec, "build", "-o", exe, srcPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("esquec compile failed: %v\n%s", err, out)
		return nil
	}

	// Time multiple executions
	start := time.Now()
	for i := 0; i < iterations; i++ {
		cmd = exec.Command(exe)
		cmd.Run() // Ignore exit code (we're just timing)
	}
	elapsed := time.Since(start)

	t.Logf("Esque %s: %d iterations in %v (%.2f µs/iter)",
		name, iterations, elapsed, float64(elapsed.Microseconds())/float64(iterations))

	return &BenchmarkResult{
		Name:       name,
		Language:   "Esque",
		Duration:   elapsed,
		Iterations: iterations,
	}
}

func printResults(t *testing.T, results map[string][]BenchmarkResult) {
	t.Log("\n" + strings.Repeat("=", 80))
	t.Log("BENCHMARK RESULTS SUMMARY")
	t.Log(strings.Repeat("=", 80))

	for bench, langs := range results {
		t.Logf("\n%s:", bench)
		t.Log(strings.Repeat("-", 40))

		var fastest time.Duration
		for _, r := range langs {
			if fastest == 0 || r.Duration < fastest {
				fastest = r.Duration
			}
		}

		for _, r := range langs {
			ratio := float64(r.Duration) / float64(fastest)
			t.Logf("  %-25s %12v  (%.2fx)", r.Language, r.Duration, ratio)
		}
	}

	t.Log("\n" + strings.Repeat("=", 80))
}
