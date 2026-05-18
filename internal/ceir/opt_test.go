package ceir

import (
	"testing"

	"github.com/esque-lang/esquec/internal/types"
)

// helper: create an int constant instruction.
func cint(id int, v int64) Inst {
	return Inst{
		Result: Value{ID: id, Type: types.I32()},
		Op:     OpConstInt,
		Imm:    v,
	}
}

// helper: create a bool constant.
func cbool(id int, v bool) Inst {
	return Inst{
		Result: Value{ID: id, Type: types.Bool()},
		Op:     OpConstBool,
		ImmB:   v,
	}
}

// helper: create a binary op.
func bin(id int, op Op, l, r int) Inst {
	return Inst{
		Result: Value{ID: id, Type: types.I32()},
		Op:     op,
		Args:   []Value{{ID: l, Type: types.I32()}, {ID: r, Type: types.I32()}},
	}
}

func TestConstFoldAddMul(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 3),
			cint(2, 4),
			bin(3, OpAdd, 1, 2),  // 3 + 4 = 7
			bin(4, OpMul, 3, 1),  // 7 * 3 = 21
		},
		Result: Value{ID: 4, Type: types.I32()},
	}
	ConstFold(fn)
	// Final inst should be OpConstInt with Imm == 21.
	last := fn.Body[len(fn.Body)-1]
	if last.Op != OpConstInt || last.Imm != 21 {
		t.Errorf("last = %+v want ConstInt 21", last)
	}
}

func TestConstFoldDivByZeroNoCrash(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 10),
			cint(2, 0),
			bin(3, OpDiv, 1, 2),
		},
		Result: Value{ID: 3, Type: types.I32()},
	}
	// Should not panic; the pass folds to 0 silently rather than failing.
	ConstFold(fn)
}

func TestConstFoldComparison(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 3),
			cint(2, 5),
			bin(3, OpLt, 1, 2),
		},
	}
	ConstFold(fn)
	last := fn.Body[len(fn.Body)-1]
	if last.Op != OpConstBool || !last.ImmB {
		t.Errorf("last = %+v want ConstBool true", last)
	}
}

func TestConstFoldSelectKnown(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cbool(1, true),
			cint(2, 100),
			cint(3, 200),
			{
				Result: Value{ID: 4, Type: types.I32()},
				Op:     OpSelect,
				Args: []Value{
					{ID: 1, Type: types.Bool()},
					{ID: 2, Type: types.I32()},
					{ID: 3, Type: types.I32()},
				},
			},
		},
	}
	ConstFold(fn)
	last := fn.Body[len(fn.Body)-1]
	if last.Op != OpCopy {
		t.Fatalf("expected OpCopy after select fold, got %v", last.Op)
	}
	if last.Args[0].ID != 2 {
		t.Errorf("select(true) didn't pick the then-branch")
	}
}

func TestConstFoldNotAndOr(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cbool(1, true),
			cbool(2, false),
			{Result: Value{ID: 3, Type: types.Bool()}, Op: OpAnd, Args: []Value{{ID: 1}, {ID: 2}}},
			{Result: Value{ID: 4, Type: types.Bool()}, Op: OpOr, Args: []Value{{ID: 1}, {ID: 2}}},
			{Result: Value{ID: 5, Type: types.Bool()}, Op: OpNot, Args: []Value{{ID: 1}}},
		},
	}
	ConstFold(fn)
	wants := map[int]bool{3: false, 4: true, 5: false}
	for _, in := range fn.Body {
		want, ok := wants[in.Result.ID]
		if !ok {
			continue
		}
		if in.Op != OpConstBool {
			t.Errorf("inst %d: op = %v want ConstBool", in.Result.ID, in.Op)
			continue
		}
		if in.ImmB != want {
			t.Errorf("inst %d: ImmB = %v want %v", in.Result.ID, in.ImmB, want)
		}
	}
}
