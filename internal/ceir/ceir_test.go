package ceir

import (
	"testing"

	"github.com/esque-lang/esquec/internal/parse"
	"github.com/esque-lang/esquec/internal/types"
)

func lower(t *testing.T, src string) *Module {
	t.Helper()
	f, err := parse.File("test.esq", []byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	cf, err := types.Check(f)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	m, err := LowerFromChecked(cf)
	if err != nil {
		t.Fatalf("lower: %v", err)
	}
	return m
}

// findFn returns the first function whose name matches; useful when
// monomorphization may produce duplicated names.
func findFn(m *Module, name string) *Func {
	for _, fn := range m.Fns {
		if fn.Name == name {
			return fn
		}
	}
	return nil
}

func countOp(fn *Func, op Op) int {
	n := 0
	for _, ins := range fn.Body {
		if ins.Op == op {
			n++
		}
	}
	return n
}

func TestLowerSimpleConst(t *testing.T) {
	m := lower(t, `fn main() -> i32 = 42`)
	fn := findFn(m, "main")
	if fn == nil {
		t.Fatalf("missing main")
	}
	if countOp(fn, OpConstInt) == 0 {
		t.Errorf("expected at least one OpConstInt; body=%v", fn.Body)
	}
}

func TestLowerArithmetic(t *testing.T) {
	m := lower(t, `fn k(x: i32, y: i32) -> i32 = x + y`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	if countOp(fn, OpAdd) == 0 {
		t.Errorf("expected OpAdd; body=%v", fn.Body)
	}
}

func TestLowerTensorAdd(t *testing.T) {
	m := lower(t, `fn k(x: f32[3], y: f32[3]) -> f32[3] = x .+ y`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	// Either OpTensorAdd or unrolled OpAdd over elements.
	if countOp(fn, OpTensorAdd)+countOp(fn, OpAdd) == 0 {
		t.Errorf("no add ops; body has %d insts", len(fn.Body))
	}
}

func TestLowerReduce(t *testing.T) {
	m := lower(t, `fn k(x: f32[4]) -> f32 = +/x`)
	fn := findFn(m, "k")
	if fn == nil {
		t.Fatal("missing k")
	}
	// Either explicit OpReduceSum or unrolled scalar additions.
	if countOp(fn, OpReduceSum)+countOp(fn, OpAdd) == 0 {
		t.Errorf("no reduce/add; body has %d insts", len(fn.Body))
	}
}

// TestLowerReduceSubChain pins the v0.14 lowering for `-/`: a scalar
// left-fold chain of OpSub, never an OpReduceSum (subtract reduce is
// non-commutative; SIMD horizontal-reduce would change rounding).
// Verified for both element types because the chain must work
// regardless of whether the existing integer or float reduce path
// would have applied.
func TestLowerReduceSubChain(t *testing.T) {
	cases := []struct {
		name, src string
		n         int // tensor length; expect n-1 OpSubs (left fold).
	}{
		{"i32", `fn k(x: i32[4]) -> i32 = -/x`, 4},
		{"f32", `fn k(x: f32[4]) -> f32 = -/x`, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := lower(t, tc.src)
			fn := findFn(m, "k")
			if fn == nil {
				t.Fatal("missing k")
			}
			if got := countOp(fn, OpSub); got != tc.n-1 {
				t.Errorf("OpSub count = %d, want %d (one per non-seed element)", got, tc.n-1)
			}
			if got := countOp(fn, OpReduceSum); got != 0 {
				t.Errorf("OpReduceSum count = %d, want 0 (non-commutative path must not use SIMD reduce)", got)
			}
		})
	}
}

// TestLowerReduceDivChain pins the v0.14 lowering for `//`: a scalar
// left-fold chain of OpDiv. Like subtract-reduce there is no
// commutative reformulation, so the SIMD reduce path must not engage.
func TestLowerReduceDivChain(t *testing.T) {
	cases := []struct {
		name, src string
		n         int
	}{
		{"i32", `fn k(x: i32[4]) -> i32 = //x`, 4},
		{"f32", `fn k(x: f32[4]) -> f32 = //x`, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := lower(t, tc.src)
			fn := findFn(m, "k")
			if fn == nil {
				t.Fatal("missing k")
			}
			if got := countOp(fn, OpDiv); got != tc.n-1 {
				t.Errorf("OpDiv count = %d, want %d (one per non-seed element)", got, tc.n-1)
			}
			if got := countOp(fn, OpReduceProd); got != 0 {
				t.Errorf("OpReduceProd count = %d, want 0 (non-commutative path must not use SIMD reduce)", got)
			}
		})
	}
}

func TestLowerCall(t *testing.T) {
	m := lower(t, `
fn id(x: i32) -> i32 = x
fn main() -> i32 = id(7)
`)
	fn := findFn(m, "main")
	if fn == nil {
		t.Fatal("missing main")
	}
	if countOp(fn, OpCall) == 0 {
		t.Errorf("expected OpCall in main; body=%v", fn.Body)
	}
}
