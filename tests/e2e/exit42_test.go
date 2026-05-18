package e2e

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TestExit42 is the v0.0 acceptance test:
//   esquec build exit42.esq -o exit42 && ./exit42; echo $? == 42
func TestExit42(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()

	// Build the esquec binary into tmp.
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}

	// Compile exit42.esq.
	src, err := filepath.Abs("exit42.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "exit42")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("expected %s to exist: %v", exe, err)
	}

	// Run the produced executable; expect exit code 42.
	runCmd := exec.Command(exe)
	err = runCmd.Run()
	if err == nil {
		t.Fatalf("expected non-zero exit, got success")
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("expected ExitError, got %T: %v", err, err)
	}
	if exitErr.ExitCode() != 42 {
		t.Fatalf("expected exit code 42, got %d", exitErr.ExitCode())
	}
}

// TestDotProduct is the v0.3 acceptance test:
// Tests tensor literals, element-wise multiply, and sum reduction.
// Computes [1.0, 2.0, 3.0] .* [4.0, 5.0, 6.0] = [4.0, 10.0, 18.0]
// Then +/ reduces to 32.0
func TestDotProduct(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()

	// Build the esquec binary into tmp.
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}

	// Compile dotproduct.esq.
	src, err := filepath.Abs("dotproduct.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "dotproduct")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("expected %s to exist: %v", exe, err)
	}

	// Run the produced executable; expect exit code 32.
	runCmd := exec.Command(exe)
	err = runCmd.Run()
	if err == nil {
		t.Fatalf("expected non-zero exit, got success")
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("expected ExitError, got %T: %v", err, err)
	}
	if exitErr.ExitCode() != 32 {
		t.Fatalf("expected exit code 32, got %d", exitErr.ExitCode())
	}
}

// TestFloatScalar tests basic f32 scalar arithmetic.
func TestFloatScalar(t *testing.T) {
	runE2ETest(t, "float_scalar.esq", "float_scalar", 4)
}

// TestTensorLiteral tests tensor literal parsing and element-wise multiplication.
func TestTensorLiteral(t *testing.T) {
	runE2ETest(t, "tensor_literal.esq", "tensor_literal", 11)
}

// TestReduceSum tests sum reduction over a tensor.
func TestReduceSum(t *testing.T) {
	runE2ETest(t, "reduce_sum.esq", "reduce_sum", 6)
}

// TestMatMul tests rank-2 tensor literals and matrix multiplication.
// Computes [[1,2],[3,4]] @ [[5,6],[7,8]] and sums all elements.
// Result: 19 + 22 + 43 + 50 = 134
func TestMatMul(t *testing.T) {
	runE2ETest(t, "matmul.esq", "matmul", 134)
}

// TestGenericDot tests shape polymorphism with a generic dot product function.
// fn dot[N: nat](a: f32[N], b: f32[N]) -> f32 = +/(a .* b)
// Computes [1,2,3] dot [4,5,6] = 4 + 10 + 18 = 32
func TestGenericDot(t *testing.T) {
	runE2ETest(t, "generic_dot.esq", "generic_dot", 32)
}

// TestAVX2Vectorize tests AVX2 vectorized 8-element tensor addition.
// Uses VADDPS (256-bit packed single) to add 8 floats in one instruction.
// Computes [1,2,3,4,5,6,7,8] .+ [1,1,1,1,1,1,1,1] = [2,3,4,5,6,7,8,9]
// Sum: 2+3+4+5+6+7+8+9 = 44
func TestAVX2Vectorize(t *testing.T) {
	runE2ETest(t, "avx2_vectorize.esq", "avx2_vectorize", 44)
}

// TestAVX2Mul tests AVX2 vectorized 8-element tensor multiplication.
// Uses VMULPS (256-bit packed single) to multiply 8 floats in one instruction.
// Computes [1,2,3,4,5,6,7,8] .* [2,2,2,2,2,2,2,2] = [2,4,6,8,10,12,14,16]
// Sum: 2+4+6+8+10+12+14+16 = 72
func TestAVX2Mul(t *testing.T) {
	runE2ETest(t, "avx2_mul.esq", "avx2_mul", 72)
}

// TestSSEVectorize tests SSE 4-element tensor addition.
// Uses ADDPS (128-bit packed single) to add 4 floats in one instruction.
// Computes [1,2,3,4] .+ [1,1,1,1] = [2,3,4,5], sum = 14
func TestSSEVectorize(t *testing.T) {
	runE2ETest(t, "sse_vectorize.esq", "sse_vectorize", 14)
}

// TestHybridVectorize tests hybrid AVX2 + SSE vectorization.
// 12 elements = 8 (AVX2) + 4 (SSE)
// Computes [1..12] .+ [1,1,...] = [2..13], sum = 90
func TestHybridVectorize(t *testing.T) {
	runE2ETest(t, "hybrid_vectorize.esq", "hybrid_vectorize", 90)
}

// TestHybridOdd tests SSE + scalar cleanup for 7 elements.
// 7 elements = 4 (SSE) + 3 (scalar)
// Computes [1..7] .* [2,2,...] = [2,4,6,8,10,12,14], sum = 56
func TestHybridOdd(t *testing.T) {
	runE2ETest(t, "hybrid_odd.esq", "hybrid_odd", 56)
}

// TestHybrid15 tests full hybrid: AVX2 + SSE + scalar.
// 15 elements = 8 (AVX2) + 4 (SSE) + 3 (scalar)
// Computes [1,1,...] .+ [1..15] = [2..16], sum = 135
func TestHybrid15(t *testing.T) {
	runE2ETest(t, "hybrid_15.esq", "hybrid_15", 135)
}

// TestFoldAdd tests the fold syntax with add: add/tensor
// Computes add/[1,2,3,4] = 1+2+3+4 = 10.
// At N=4 this exercises the SSE3 horizontal-SIMD reduction path
// (haddps; haddps).
func TestFoldAdd(t *testing.T) {
	runE2ETest(t, "fold_add.esq", "fold_add", 10)
}

// TestFoldAdd8 tests sum reduction at exactly 8 elements.
// Computes add/[1..8] = 36. This exercises the AVX2 horizontal-SIMD
// reduction path: vmovups ymm; vextractf128; addps; haddps; haddps.
func TestFoldAdd8(t *testing.T) {
	runE2ETest(t, "fold_add_8.esq", "fold_add_8", 36)
}

// TestFoldAdd16 tests sum reduction at 16 elements (strip-mined).
// Computes add/[1..16] = 136. Exercises (a) the unrolled YMM
// accumulator chain (load256+addps256 over two chunks) plus horizontal
// reduce, and (b) the OpConstFloat spill path because 16 simultaneous
// scalar literals exceed the 15-slot SIMD register pool.
func TestFoldAdd16(t *testing.T) {
	runE2ETest(t, "fold_add_16.esq", "fold_add_16", 136)
}

// TestFoldAdd11 tests sum reduction at 11 elements (8 SIMD + 3 scalar
// tail). Computes add/[1..11] = 66.
func TestFoldAdd11(t *testing.T) {
	runE2ETest(t, "fold_add_11.esq", "fold_add_11", 66)
}

// TestFoldAdd7 tests sum reduction at 7 elements: SSE3 4-lane prefix
// (Load128 + 2x HAddPS128) plus a 3-element scalar tail. Closes the
// 4 < N < 8 gap end-to-end. Computes add/[1..7] = 28.
func TestFoldAdd7(t *testing.T) {
	runE2ETest(t, "fold_add_7.esq", "fold_add_7", 28)
}


// TestFoldMul tests the fold syntax with mul: mul/tensor
// Computes mul/[1,2,3,4] = 1*2*3*4 = 24
func TestFoldMul(t *testing.T) {
	runE2ETest(t, "fold_mul.esq", "fold_mul", 24)
}

// TestLambdaFold tests fold with an inline lambda: (|a, b| a + b) / tensor
// Computes ((1+2)+3)+4 = 10
func TestLambdaFold(t *testing.T) {
	runE2ETest(t, "lambda_fold.esq", "lambda_fold", 10)
}

// TestIterateLambda tests iterate with an inline lambda: iterate(n, init, |x| x * 2.0)
// Computes 1.0 * 2^5 = 32
func TestIterateLambda(t *testing.T) {
	runE2ETest(t, "iterate_lambda.esq", "iterate_lambda", 32)
}

// TestLambdaFoldMul tests fold with multiplication lambda
// Computes ((1*2)*3)*4 = 24
func TestLambdaFoldMul(t *testing.T) {
	runE2ETest(t, "lambda_fold_mul.esq", "lambda_fold_mul", 24)
}

// TestIterateAdd tests iterate with addition lambda
// Computes 0 + 5*10 = 50
func TestIterateAdd(t *testing.T) {
	runE2ETest(t, "iterate_add.esq", "iterate_add", 50)
}

// TestPrintI32 verifies the runtime print_i32 intrinsic. The program
// `print_i32(42)` should write "42\n" to stdout and return 42, which
// main hands back to the kernel as the exit code.
func TestPrintI32(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs("print_i32.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "print_i32")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	runCmd := exec.Command(exe)
	stdout, err := runCmd.Output()
	if err == nil {
		t.Fatalf("expected non-zero exit, got success; stdout=%q", stdout)
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("expected ExitError, got %T: %v", err, err)
	}
	if exitErr.ExitCode() != 42 {
		t.Fatalf("exit code = %d, want 42 (stdout=%q)", exitErr.ExitCode(), stdout)
	}
	if got, want := string(stdout), "42\n"; got != want {
		t.Errorf("stdout = %q, want %q", got, want)
	}
}

// TestPrintF32 verifies the runtime print_f32 intrinsic. The program
// prints three formatted floats and returns 7. We assert both the
// exit code and the exact stdout, which exercises the
// integer-and-fractional split, the leading-zero fractional padding,
// and the leading-'-' sign handling for negatives.
func TestPrintF32(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs("print_f32.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "print_f32")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	runCmd := exec.Command(exe)
	stdout, err := runCmd.Output()
	if err == nil {
		t.Fatalf("expected non-zero exit, got success; stdout=%q", stdout)
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("expected ExitError, got %T: %v", err, err)
	}
	if exitErr.ExitCode() != 7 {
		t.Fatalf("exit code = %d, want 7 (stdout=%q)", exitErr.ExitCode(), stdout)
	}
	want := "3.140000\n0.000000\n-2.500000\n"
	if got := string(stdout); got != want {
		t.Errorf("stdout = %q, want %q", got, want)
	}
}

// runE2EStdoutTest compiles `srcFile`, runs the resulting executable,
// and asserts that it exits 0 and prints exactly `wantStdout`.
func runE2EStdoutTest(t *testing.T, srcFile, exeName, wantStdout string) {
	t.Helper()
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs(srcFile)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, exeName)
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}
	stdout, err := exec.Command(exe).Output()
	if err != nil {
		t.Fatalf("run %s: %v (stdout=%q)", exeName, err, stdout)
	}
	if got := string(stdout); got != wantStdout {
		t.Errorf("stdout = %q, want %q", got, wantStdout)
	}
}

// TestPrintStr verifies the runtime print_str intrinsic on a literal.
func TestPrintStr(t *testing.T) {
	runE2EStdoutTest(t, "print_str.esq", "print_str", "hello, world\n")
}

// TestPrintStrLetBind verifies that a string literal stored in a let
// binding round-trips through CEIR/MIR and reaches print_str at the
// use site with the correct (ptr, len) pair.
func TestPrintStrLetBind(t *testing.T) {
	runE2EStdoutTest(t, "print_str_letbind.esq", "print_str_letbind", "esque\n")
}

// TestPrintStrEscapes verifies that lex-time escape decoding survives
// rodata interning and lands in the output stream as the expected
// bytes.
func TestPrintStrEscapes(t *testing.T) {
	runE2EStdoutTest(t, "print_str_escapes.esq", "print_str_escapes", "a\tb\nc\\d\n")
}

// TestRangeSum checks +/(0..5) = 10.
func TestRangeSum(t *testing.T) {
	runE2ETest(t, "range_sum.esq", "range_sum", 10)
}

// TestRangeInclusive checks +/(1..=5) = 15.
func TestRangeInclusive(t *testing.T) {
	runE2ETest(t, "range_inclusive.esq", "range_inclusive", 15)
}

// TestRangePipeline checks (0..5) |> sum = 10. Exercises the range
// expression as the LHS of a pipeline into a named function.
func TestRangePipeline(t *testing.T) {
	runE2ETest(t, "range_pipeline.esq", "range_pipeline", 10)
}

// TestTabulateSquares checks +/(tabulate(5, |i| i*i)) = 0+1+4+9+16 = 30.
func TestTabulateSquares(t *testing.T) {
	runE2ETest(t, "tabulate_squares.esq", "tabulate_squares", 30)
}

// TestScanPrefix checks +/(scan(0, |a,x| a+x, [1,2,3,4])) = 20
// (running prefix sums [1,3,6,10] reduce to 20).
func TestScanPrefix(t *testing.T) {
	runE2ETest(t, "scan_prefix.esq", "scan_prefix", 20)
}

// TestIterateUntilSimple checks iterate_until(0, |s| s+1, |s| s==7, 10) = 7.
func TestIterateUntilSimple(t *testing.T) {
	runE2ETest(t, "iterate_until_simple.esq", "iterate_until_simple", 7)
}

// TestLargeTabulate exercises the runtime tabulate-loop form by asking
// for an N=100 result, well above the unroll ceiling. Computes
// +/(tabulate(100, |i| i)) % 256 = 4950 % 256 = 86.
func TestLargeTabulate(t *testing.T) {
	runE2ETest(t, "large_tabulate.esq", "large_tabulate", 86)
}

// TestF64Arith exercises the v0.12 f64 scalar codegen path: 8-byte
// constant loads via MOVSD, ADDSD/SUBSD/MULSD/DIVSD, and CVTSI2SD /
// CVTTSD2SI for i32 <-> f64 casts. Computes
//   ((17 as f64 * 2.5 - 0.5) / 1.0) as i32 = 42.
func TestF64Arith(t *testing.T) {
	runE2ETest(t, "f64_arith.esq", "f64_arith", 42)
}

// TestLetIn exercises the v0.12 `let x = e in body` expression form.
// Three nested let-ins build a = 10, b = 20, c = a+b, returning
// c + 12 = 42.
func TestLetIn(t *testing.T) {
	runE2ETest(t, "let_in.esq", "let_in", 42)
}

// TestConstShape exercises the v0.11 const-eval pass over shape
// arithmetic: range bounds, tabulate N, and iterate_until max are all
// supplied as const-foldable arithmetic expressions rather than bare
// integer literals. Computes
//   +/(0..(2+3))                                 = 10
//   +/(tabulate(2*16+1, |i| i))                  = 528
//   iterate_until(0, |s| s+1, |s| s == 7, 5*2)   =   7
//   sum                                          = 545
//   545 % 256                                    =  33
func TestConstShape(t *testing.T) {
	runE2ETest(t, "const_shape.esq", "const_shape", 33)
}

// TestLargeScan exercises the runtime scan-loop form chained with the
// runtime tabulate-loop form. scan(0, +, tabulate(50, |i| 1)) produces
// prefix sums 1..50; +/ those = 1275; 1275 % 256 = 251.
func TestLargeScan(t *testing.T) {
	runE2ETest(t, "large_scan.esq", "large_scan", 251)
}

// TestLargeEachPrint exercises the runtime each-loop form by walking a
// 33-element range and verifying every print_i32 fires in order.
func TestLargeEachPrint(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs("large_each.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "large_each")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	runCmd := exec.Command(exe)
	stdout, err := runCmd.Output()
	if err != nil {
		t.Fatalf("expected success, got error: %v (stdout=%q)", err, stdout)
	}
	want := ""
	for i := 0; i < 33; i++ {
		want += fmt.Sprintf("%d\n", i)
	}
	if got := string(stdout); got != want {
		t.Errorf("stdout = %q, want %q", got, want)
	}
}

// TestEachPrint checks `each(0..5, print_i32); 0` prints "0\n1\n2\n3\n4\n"
// and exits 0.
func TestEachPrint(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs("each_print.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "each_print")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	runCmd := exec.Command(exe)
	stdout, err := runCmd.Output()
	if err != nil {
		t.Fatalf("expected success, got error: %v (stdout=%q)", err, stdout)
	}
	want := "0\n1\n2\n3\n4\n"
	if got := string(stdout); got != want {
		t.Errorf("stdout = %q, want %q", got, want)
	}
}

// TestEachUserIOFn drives v0.13: the legacy each-allowlist is gone,
// so a user-defined `@io` wrapper around print_i32 is a valid each
// callee. Output is one number per line for 0..3.
func TestEachUserIOFn(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs("each_user_io.esq")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, "each_user_io")
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	runCmd := exec.Command(exe)
	stdout, err := runCmd.Output()
	if err != nil {
		t.Fatalf("expected success, got error: %v (stdout=%q)", err, stdout)
	}
	want := "0\n1\n2\n"
	if got := string(stdout); got != want {
		t.Errorf("stdout = %q, want %q", got, want)
	}
}

// TestEffectMismatchRejected drives v0.13: a pure main calling an @io
// function fails the type checker, with a message that mentions @io.
func TestEffectMismatchRejected(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	if out, err := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec").CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	srcPath := filepath.Join(tmp, "bad.esq")
	if err := os.WriteFile(srcPath, []byte("fn main() -> i32 = print_i32(1)\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	cmd := exec.Command(esquec, "check", srcPath)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected check failure, got success: %s", out)
	}
	if want := "@io"; !contains(string(out), want) {
		t.Fatalf("expected diagnostic to mention %q, got: %s", want, out)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// TestCharLiteral checks that 'A' is an i32-typed integer literal whose
// value is the codepoint 65.
func TestCharLiteral(t *testing.T) {
	runE2ETest(t, "char_literal.esq", "char_literal", 65)
}

// TestIntSuffixI32 checks that an explicit _i32 suffix round-trips
// through parser → type-check → codegen.
func TestIntSuffixI32(t *testing.T) {
	runE2ETest(t, "int_suffix_i32.esq", "int_suffix_i32", 99)
}

// TestIntSuffixI64 (v0.11) exercises the 64-bit GPR path: parse → check
// → CEIR (carrying types.I64) → MIR → REX.W-prefixed encoder forms.
func TestIntSuffixI64(t *testing.T) {
	runE2ETest(t, "int_suffix_i64.esq", "int_suffix_i64", 42)
}

// TestI64Arith (v0.11) verifies that 64-bit IMUL is actually being
// emitted: 10^12 mod 256 = 0 only if the multiply did not silently
// truncate to 32 bits. Adds 71 so the result is in the exit-code byte.
func TestI64Arith(t *testing.T) {
	runE2ETest(t, "i64_arith.esq", "i64_arith", 71)
}

// TestI64SignedDiv (v0.11) covers IDIV+CQO with a negative dividend.
// -100/7 = -14 (rounded toward zero), -14 + 56 = 42.
func TestI64SignedDiv(t *testing.T) {
	runE2ETest(t, "i64_signed_div.esq", "i64_signed_div", 42)
}

// TestU32Div (v0.11) pins down unsigned DIV: a dividend > 2^31 must
// produce a positive quotient. Signed IDIV would either trap or give
// a nonsense answer for 4_000_000_000 / 100_000_000.
func TestU32Div(t *testing.T) {
	runE2ETest(t, "u32_div.esq", "u32_div", 40)
}

// TestU32Compare (v0.11) pins down unsigned ordering: 3_000_000_000
// is "greater than" 100 only under unsigned semantics (CCA), so this
// returning 7 (and not 99) proves SETcc dispatches on operand
// signedness.
func TestU32Compare(t *testing.T) {
	runE2ETest(t, "u32_compare.esq", "u32_compare", 7)
}

// TestI8Arith (v0.12): basic i8 subtraction with both operands as
// i8 literals. (100_i8 - 58_i8) = 42_i8, widened to 42_i32.
func TestI8Arith(t *testing.T) {
	runE2ETest(t, "i8_arith.esq", "i8_arith", 42)
}

// TestI8Negative (v0.12): exercises the canonical sign-extended
// representation by adding -10_i8 + 52_i8.
func TestI8Negative(t *testing.T) {
	runE2ETest(t, "i8_negative.esq", "i8_negative", 42)
}

// TestU8Arith (v0.12): exit 250. 200_u8 + 50_u8 = 250 with no wrap.
func TestU8Arith(t *testing.T) {
	runE2ETest(t, "u8_arith.esq", "u8_arith", 250)
}

// TestU8Div (v0.12) verifies the unsigned DIV path canonicalises u8
// results: 200/4 = 50.
func TestU8Div(t *testing.T) {
	runE2ETest(t, "u8_div.esq", "u8_div", 50)
}

// TestI8Compare (v0.12) verifies signed less-than for i8: -1 < 0.
func TestI8Compare(t *testing.T) {
	runE2ETest(t, "i8_compare.esq", "i8_compare", 42)
}

// TestU8Compare (v0.12) verifies unsigned greater-than for u8:
// 200 > 100 under unsigned semantics.
func TestU8Compare(t *testing.T) {
	runE2ETest(t, "u8_compare.esq", "u8_compare", 42)
}

// TestI8CastRoundtrip (v0.12) verifies that i32→i8→i32 truncates the
// upper bytes and sign-extends the low byte: 300 -> 0x2C = 44.
func TestI8CastRoundtrip(t *testing.T) {
	runE2ETest(t, "i8_cast_roundtrip.esq", "i8_cast_roundtrip", 44)
}

// TestU8CastRoundtrip (v0.12) verifies the unsigned variant of the
// truncate-then-widen round trip: 300 -> 0x2C = 44.
func TestU8CastRoundtrip(t *testing.T) {
	runE2ETest(t, "u8_cast_roundtrip.esq", "u8_cast_roundtrip", 44)
}

// TestI8Arg (v0.12) verifies that the function-entry canonicalisation
// pass sign-extends i8 parameters before the body sees them.
func TestI8Arg(t *testing.T) {
	runE2ETest(t, "i8_arg.esq", "i8_arg", 42)
}

// TestF32AcrossCall locks in the v0.13 fix for caller-saved SIMD
// registers across @io calls: an f32 SSA value used after print_f32
// must be saved/restored around the call. a=2.0 → print "2.000000"
// → 2.0 * 21.0 → exit 42.
func TestF32AcrossCall(t *testing.T) {
	runE2EStdoutExitTest(t, "f32_across_call.esq", "f32_across_call", "2.000000\n", 42)
}

// TestTensorAcrossCall is the GPR companion to TestF32AcrossCall: it
// forces a rodata pointer in a caller-saved register to stay live
// across an @io call. Catches the defined[Result.ID==0] sink-op
// regression that hid xs from the liveAcrossCall analysis.
func TestTensorAcrossCall(t *testing.T) {
	runE2EStdoutExitTest(t, "tensor_across_call.esq", "tensor_across_call", "1.000000\n", 142)
}

// runE2EStdoutExitTest compiles `srcFile`, runs it, and asserts both
// the stdout and the exit code. Used for tests that need to verify
// runtime intrinsic output alongside a non-zero exit.
func runE2EStdoutExitTest(t *testing.T, srcFile, exeName, wantStdout string, wantExit int) {
	t.Helper()
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	tmp := t.TempDir()
	esquec := filepath.Join(tmp, "esquec")
	if out, err := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec").CombinedOutput(); err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}
	src, err := filepath.Abs(srcFile)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, exeName)
	if out, err := exec.Command(esquec, "build", "-o", exe, src).CombinedOutput(); err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}
	stdout, err := exec.Command(exe).Output()
	if wantExit == 0 {
		if err != nil {
			t.Fatalf("expected exit 0, got %v (stdout=%q)", err, stdout)
		}
	} else {
		if err == nil {
			t.Fatalf("expected exit %d, got success (stdout=%q)", wantExit, stdout)
		}
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("expected ExitError, got %T: %v", err, err)
		}
		if exitErr.ExitCode() != wantExit {
			t.Fatalf("exit code = %d, want %d (stdout=%q)", exitErr.ExitCode(), wantExit, stdout)
		}
	}
	if got := string(stdout); got != wantStdout {
		t.Errorf("stdout = %q, want %q", got, wantStdout)
	}
}

// runE2ETest is a helper that compiles and runs a test program, checking the exit code.
func runE2ETest(t *testing.T, srcFile, exeName string, expectedExitCode int) {
	t.Helper()

	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	tmp := t.TempDir()

	// Build the esquec binary into tmp.
	esquec := filepath.Join(tmp, "esquec")
	cmd := exec.Command("go", "build", "-o", esquec, "../../cmd/esquec")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build esquec: %v\n%s", err, out)
	}

	// Compile the source file.
	src, err := filepath.Abs(srcFile)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	exe := filepath.Join(tmp, exeName)
	cmd = exec.Command(esquec, "build", "-o", exe, src)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("esquec build: %v\n%s", err, out)
	}

	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("expected %s to exist: %v", exe, err)
	}

	// Run the produced executable.
	runCmd := exec.Command(exe)
	err = runCmd.Run()

	if expectedExitCode == 0 {
		if err != nil {
			t.Fatalf("expected success, got error: %v", err)
		}
		return
	}

	if err == nil {
		t.Fatalf("expected non-zero exit, got success")
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("expected ExitError, got %T: %v", err, err)
	}
	if exitErr.ExitCode() != expectedExitCode {
		t.Fatalf("expected exit code %d, got %d", expectedExitCode, exitErr.ExitCode())
	}
}
