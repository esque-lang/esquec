package ceir

import (
	"testing"

	"github.com/esque-lang/esquec/internal/types"
)

func findResult(fn *Func, id int) *Inst {
	for i := range fn.Body {
		if fn.Body[i].Result.ID == id {
			return &fn.Body[i]
		}
	}
	return nil
}

func TestSimplifyAddZero(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 5),
			cint(2, 0),
			bin(3, OpAdd, 1, 2),
		},
	}
	Simplify(fn)
	r := findResult(fn, 3)
	if r.Op != OpCopy || r.Args[0].ID != 1 {
		t.Errorf("x + 0 not simplified to copy(x): %+v", r)
	}
}

func TestSimplifyZeroAdd(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 0),
			cint(2, 9),
			bin(3, OpAdd, 1, 2),
		},
	}
	Simplify(fn)
	r := findResult(fn, 3)
	if r.Op != OpCopy || r.Args[0].ID != 2 {
		t.Errorf("0 + x not simplified to copy(x): %+v", r)
	}
}

func TestSimplifySubSelfToZero(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			{Result: Value{ID: 1, Type: types.I32()}, Op: OpCopy, Args: []Value{{ID: 99}}},
			bin(3, OpSub, 1, 1),
		},
	}
	Simplify(fn)
	r := findResult(fn, 3)
	if r.Op != OpConstInt || r.Imm != 0 {
		t.Errorf("x - x not simplified to const 0: %+v", r)
	}
}

func TestSimplifyMulOne(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 1),
			cint(2, 7),
			bin(3, OpMul, 1, 2),
		},
	}
	Simplify(fn)
	r := findResult(fn, 3)
	if r.Op != OpCopy || r.Args[0].ID != 2 {
		t.Errorf("1 * x: %+v", r)
	}
}

func TestSimplifyMulZero(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			{Result: Value{ID: 1, Type: types.I32()}, Op: OpCopy, Args: []Value{{ID: 99}}},
			cint(2, 0),
			bin(3, OpMul, 1, 2),
		},
	}
	Simplify(fn)
	r := findResult(fn, 3)
	if r.Op != OpConstInt || r.Imm != 0 {
		t.Errorf("x * 0 -> const 0: %+v", r)
	}
}

func TestSimplifyDivOne(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			{Result: Value{ID: 1, Type: types.I32()}, Op: OpCopy, Args: []Value{{ID: 99}}},
			cint(2, 1),
			bin(3, OpDiv, 1, 2),
		},
	}
	Simplify(fn)
	r := findResult(fn, 3)
	if r.Op != OpCopy || r.Args[0].ID != 1 {
		t.Errorf("x / 1: %+v", r)
	}
}

func TestSimplifyAndTrueFalse(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			{Result: Value{ID: 1, Type: types.Bool()}, Op: OpCopy, Args: []Value{{ID: 99}}},
			cbool(2, true),
			cbool(3, false),
			{Result: Value{ID: 4, Type: types.Bool()}, Op: OpAnd, Args: []Value{{ID: 1}, {ID: 2}}},
			{Result: Value{ID: 5, Type: types.Bool()}, Op: OpAnd, Args: []Value{{ID: 1}, {ID: 3}}},
		},
	}
	Simplify(fn)
	r4 := findResult(fn, 4)
	r5 := findResult(fn, 5)
	if r4.Op != OpCopy || r4.Args[0].ID != 1 {
		t.Errorf("x && true: %+v", r4)
	}
	if r5.Op != OpConstBool || r5.ImmB {
		t.Errorf("x && false: %+v", r5)
	}
}

func TestSimplifyOrTrueFalse(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			{Result: Value{ID: 1, Type: types.Bool()}, Op: OpCopy, Args: []Value{{ID: 99}}},
			cbool(2, true),
			cbool(3, false),
			{Result: Value{ID: 4, Type: types.Bool()}, Op: OpOr, Args: []Value{{ID: 1}, {ID: 2}}},
			{Result: Value{ID: 5, Type: types.Bool()}, Op: OpOr, Args: []Value{{ID: 1}, {ID: 3}}},
		},
	}
	Simplify(fn)
	r4 := findResult(fn, 4)
	r5 := findResult(fn, 5)
	if r4.Op != OpConstBool || !r4.ImmB {
		t.Errorf("x || true: %+v", r4)
	}
	if r5.Op != OpCopy || r5.Args[0].ID != 1 {
		t.Errorf("x || false: %+v", r5)
	}
}

func TestSimplifyIdempotent(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 0),
			{Result: Value{ID: 2, Type: types.I32()}, Op: OpCopy, Args: []Value{{ID: 99}}},
			bin(3, OpAdd, 2, 1),
		},
	}
	Simplify(fn)
	before := len(fn.Body)
	r1 := findResult(fn, 3)
	op1, src1 := r1.Op, r1.Args[0].ID
	Simplify(fn)
	if len(fn.Body) != before {
		t.Fatalf("len changed: %d -> %d", before, len(fn.Body))
	}
	r2 := findResult(fn, 3)
	if r2.Op != op1 || r2.Args[0].ID != src1 {
		t.Errorf("not idempotent")
	}
}
