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
