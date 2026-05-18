package mir

import (
	"testing"

	"github.com/esque-lang/esquec/internal/ceir"
	"github.com/esque-lang/esquec/internal/parse"
	"github.com/esque-lang/esquec/internal/types"
)

func lowerMIR(t *testing.T, src string) *Module {
	t.Helper()
	f, err := parse.File("test.esq", []byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	cf, err := types.Check(f)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	cm, err := ceir.LowerFromChecked(cf)
	if err != nil {
		t.Fatalf("ceir lower: %v", err)
	}
	return LowerFromCEIR(cm)
}

func findFn(m *Module, name string) *Func {
	for _, fn := range m.Fns {
		if fn.Name == name {
			return fn
		}
	}
	return nil
}

func TestMIRBasic(t *testing.T) {
	m := lowerMIR(t, `fn main() -> i32 = 42`)
	fn := findFn(m, "main")
	if fn == nil {
		t.Fatalf("no main; fns=%v", funcNames(m))
	}
	if len(fn.Body) == 0 {
		t.Errorf("empty body")
	}
}

func TestMIRArithmetic(t *testing.T) {
	m := lowerMIR(t, `fn k(x: i32, y: i32) -> i32 = x + y`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	if len(fn.Body) == 0 {
		t.Error("empty body")
	}
}

func TestMIRTensorAdd(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32[3], y: f32[3]) -> f32[3] = x .+ y`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	if len(fn.Body) == 0 {
		t.Error("empty body")
	}
}

// TestMIRReduceSum8SIMDPath verifies that a rank-1 sum reduction over
// exactly 8 elements lowers to the AVX2 horizontal-SIMD sequence
// (Load256 + ExtractF128Lo/Hi + AddPS128 + 2x HAddPS128).
func TestMIRReduceSum8SIMDPath(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32[8]) -> f32 = add/x`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	want := []Op{OpLoad256, OpExtractF128Lo, OpExtractF128Hi, OpAddPS128, OpHAddPS128, OpHAddPS128}
	for _, w := range want {
		found := false
		for _, in := range fn.Body {
			if in.Op == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing op %v in N=8 reduce-sum lowering; body=%v", w, opNames(fn))
		}
	}
}

// TestMIRReduceSum4SIMDPath verifies that a rank-1 sum reduction over
// exactly 4 elements lowers to the SSE3 horizontal-SIMD sequence
// (Load128 + 2x HAddPS128).
func TestMIRReduceSum4SIMDPath(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32[4]) -> f32 = add/x`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	want := []Op{OpLoad128, OpHAddPS128}
	for _, w := range want {
		found := false
		for _, in := range fn.Body {
			if in.Op == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing op %v in N=4 reduce-sum lowering; body=%v", w, opNames(fn))
		}
	}
}

// TestMIRReduceSum16StripMined verifies that a rank-1 sum reduction over
// 16 elements (a multiple of 8) emits a strip-mined YMM accumulator:
// two OpLoad256 chunks, one OpAddPS256 to combine them, then the
// horizontal-reduce tail (extract+addps+haddps+haddps).
func TestMIRReduceSum16StripMined(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32[16]) -> f32 = add/x`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	count := func(op Op) int {
		n := 0
		for _, in := range fn.Body {
			if in.Op == op {
				n++
			}
		}
		return n
	}
	if got := count(OpLoad256); got != 2 {
		t.Errorf("OpLoad256 count = %d, want 2", got)
	}
	if got := count(OpAddPS256); got != 1 {
		t.Errorf("OpAddPS256 count = %d, want 1 (one accumulator step for 2 chunks)", got)
	}
	if got := count(OpExtractF128Lo); got != 1 {
		t.Errorf("OpExtractF128Lo count = %d, want 1", got)
	}
	if got := count(OpExtractF128Hi); got != 1 {
		t.Errorf("OpExtractF128Hi count = %d, want 1", got)
	}
	if got := count(OpHAddPS128); got != 2 {
		t.Errorf("OpHAddPS128 count = %d, want 2", got)
	}
}

// TestMIRReduceSum24StripMined verifies the strip-mined accumulator at
// 24 elements (3 chunks): three OpLoad256 and two OpAddPS256.
func TestMIRReduceSum24StripMined(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32[24]) -> f32 = add/x`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	count := func(op Op) int {
		n := 0
		for _, in := range fn.Body {
			if in.Op == op {
				n++
			}
		}
		return n
	}
	if got := count(OpLoad256); got != 3 {
		t.Errorf("OpLoad256 count = %d, want 3", got)
	}
	if got := count(OpAddPS256); got != 2 {
		t.Errorf("OpAddPS256 count = %d, want 2", got)
	}
}

// TestMIRReduceSum10WithTail verifies that a non-multiple-of-8 size
// emits the SIMD body for the 8-aligned prefix plus a scalar tail
// (OpLoad32 + OpAddF32) for the remaining 2 elements.
func TestMIRReduceSum10WithTail(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32[10]) -> f32 = add/x`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	count := func(op Op) int {
		n := 0
		for _, in := range fn.Body {
			if in.Op == op {
				n++
			}
		}
		return n
	}
	if got := count(OpLoad256); got != 1 {
		t.Errorf("OpLoad256 count = %d, want 1 (one SIMD chunk)", got)
	}
	// Two scalar tail elements: 2x OpLoad32 + 2x OpAddF32.
	if got := count(OpLoad32); got != 2 {
		t.Errorf("OpLoad32 count = %d, want 2 (scalar tail)", got)
	}
	if got := count(OpAddF32); got != 2 {
		t.Errorf("OpAddF32 count = %d, want 2 (scalar tail)", got)
	}
}

// TestMIRReduceSum5To7SSE3WithTail verifies that 5 <= N < 8 takes the
// SSE3 path (Load128 + 2x HAddPS128) plus a scalar tail of N-4 elements.
// Three sub-cases verify N=5, 6, 7 each emit the expected counts.
func TestMIRReduceSum5To7SSE3WithTail(t *testing.T) {
	cases := []struct {
		n        int
		wantTail int
	}{
		{5, 1},
		{6, 2},
		{7, 3},
	}
	for _, tc := range cases {
		src := "fn k(x: f32[" + itoa(tc.n) + "]) -> f32 = add/x"
		m := lowerMIR(t, src)
		fn := findFn(m, "k")
		if fn == nil {
			t.Fatalf("N=%d: missing k", tc.n)
		}
		count := func(op Op) int {
			c := 0
			for _, in := range fn.Body {
				if in.Op == op {
					c++
				}
			}
			return c
		}
		if got := count(OpLoad128); got != 1 {
			t.Errorf("N=%d: OpLoad128 count = %d, want 1", tc.n, got)
		}
		if got := count(OpHAddPS128); got != 2 {
			t.Errorf("N=%d: OpHAddPS128 count = %d, want 2", tc.n, got)
		}
		if got := count(OpLoad32); got != tc.wantTail {
			t.Errorf("N=%d: OpLoad32 count = %d, want %d", tc.n, got, tc.wantTail)
		}
		if got := count(OpAddF32); got != tc.wantTail {
			t.Errorf("N=%d: OpAddF32 count = %d, want %d", tc.n, got, tc.wantTail)
		}
		// N>=8-only ops should NOT appear.
		if got := count(OpLoad256); got != 0 {
			t.Errorf("N=%d: OpLoad256 count = %d, want 0", tc.n, got)
		}
	}
}

// TestMIRRodataConstTensor verifies the .rodata fast path for a
// fully-constant rank-1 tensor literal: the OpStackAlloc + per-element
// OpStore32 sequence is replaced by a single OpRodataPtr, the
// dead OpConstFloat materializations are elided, and the module
// records exactly one RodataEntry with the flattened bytes.
func TestMIRRodataConstTensor(t *testing.T) {
	m := lowerMIR(t, `fn main() -> i32 = {
		let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
		           9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
		(add/arr) as i32
	}`)
	fn := findFn(m, "main")
	if fn == nil {
		t.Fatalf("no main; fns=%v", funcNames(m))
	}
	count := func(op Op) int {
		n := 0
		for _, in := range fn.Body {
			if in.Op == op {
				n++
			}
		}
		return n
	}
	if got := count(OpRodataPtr); got != 1 {
		t.Errorf("OpRodataPtr count = %d, want 1", got)
	}
	// The 16 OpConstFloat values that fed the literal must be elided —
	// they were absorbed into the rodata blob and have no surviving use.
	if got := count(OpConstFloat); got != 0 {
		t.Errorf("OpConstFloat count = %d, want 0 (absorbed into .rodata)", got)
	}
	// And the per-element store sequence must not materialize.
	if got := count(OpStore32); got != 0 {
		t.Errorf("OpStore32 count = %d, want 0 (no per-element init for rodata tensors)", got)
	}
	if len(m.Rodata) != 1 {
		t.Fatalf("Rodata entries = %d, want 1", len(m.Rodata))
	}
	if got := len(m.Rodata[0].Bytes); got != 16*4 {
		t.Errorf("Rodata bytes = %d, want %d", got, 16*4)
	}
}

// TestMIRRodataNonConstTensorFallsBack verifies that a tensor literal
// containing a non-constant element (here a function parameter) does
// NOT take the rodata path: no Rodata entry is recorded and the
// stack-store sequence remains.
func TestMIRRodataNonConstTensorFallsBack(t *testing.T) {
	m := lowerMIR(t, `fn k(x: f32) -> f32 = {
		let arr = [x, 2.0, 3.0, 4.0];
		add/arr
	}`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatalf("no k; fns=%v", funcNames(m))
	}
	rodataPtrs := 0
	stackAllocs := 0
	for _, in := range fn.Body {
		if in.Op == OpRodataPtr {
			rodataPtrs++
		}
		if in.Op == OpStackAlloc {
			stackAllocs++
		}
	}
	if rodataPtrs != 0 {
		t.Errorf("OpRodataPtr count = %d, want 0 (literal contains non-const)", rodataPtrs)
	}
	if stackAllocs == 0 {
		t.Errorf("expected at least one OpStackAlloc for non-const tensor literal")
	}
	if len(m.Rodata) != 0 {
		t.Errorf("Rodata entries = %d, want 0 (no constant tensor)", len(m.Rodata))
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

func opNames(fn *Func) []Op {
	var out []Op
	for _, in := range fn.Body {
		out = append(out, in.Op)
	}
	return out
}

func funcNames(m *Module) []string {
	var out []string
	for _, fn := range m.Fns {
		out = append(out, fn.Name)
	}
	return out
}
