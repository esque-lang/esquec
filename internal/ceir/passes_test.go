package ceir

import (
	"testing"

	"github.com/esque-lang/esquec/internal/types"
)

func TestDCERemovesUnusedConst(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 1),  // unused
			cint(2, 2),  // used as result
			cint(3, 99), // unused
		},
		Result: Value{ID: 2, Type: types.I32()},
	}
	DCE(fn)
	if len(fn.Body) != 1 {
		t.Fatalf("body len = %d, want 1; body=%+v", len(fn.Body), fn.Body)
	}
	if fn.Body[0].Result.ID != 2 {
		t.Errorf("kept wrong inst: %+v", fn.Body[0])
	}
}

func TestDCEKeepsSideEffectingCalls(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 7),
			{
				Result: Value{ID: 2, Type: types.I32()},
				Op:     OpCall,
				FnName: "side_effect",
				Args:   []Value{{ID: 1, Type: types.I32()}},
			},
			cint(3, 0),
		},
		Result: Value{ID: 3, Type: types.I32()},
	}
	DCE(fn)
	// Call and its operand and the result should remain.
	hasCall := false
	for _, in := range fn.Body {
		if in.Op == OpCall {
			hasCall = true
		}
	}
	if !hasCall {
		t.Errorf("DCE removed side-effecting call; body=%+v", fn.Body)
	}
}

func TestCSEDeduplicatesAdds(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 3),
			cint(2, 4),
			bin(3, OpAdd, 1, 2),  // 3+4
			bin(4, OpAdd, 1, 2),  // duplicate of 3
			bin(5, OpMul, 3, 4),  // uses both
		},
		Result: Value{ID: 5, Type: types.I32()},
	}
	CSE(fn)
	// inst 4 should now be a copy of value 3.
	var got Inst
	for _, in := range fn.Body {
		if in.Result.ID == 4 {
			got = in
		}
	}
	if got.Op != OpCopy {
		t.Errorf("inst 4 op = %v want OpCopy; body=%+v", got.Op, fn.Body)
	}
	if got.Args[0].ID != 3 {
		t.Errorf("inst 4 copy source = %d want 3", got.Args[0].ID)
	}
}

func TestPipelineIdempotent(t *testing.T) {
	fn := &Func{
		Body: []Inst{
			cint(1, 1),
			cint(2, 2),
			bin(3, OpAdd, 1, 2),  // -> 3
			bin(4, OpAdd, 1, 2),  // duplicate
			bin(5, OpAdd, 3, 4),  // 3+3 = 6
		},
		Result: Value{ID: 5, Type: types.I32()},
	}
	pipe := DefaultPipeline()
	for _, p := range pipe {
		p.Run(fn)
	}
	// Final should be a single OpConstInt with the folded value 6.
	if len(fn.Body) == 0 {
		t.Fatal("body empty after pipeline")
	}
	last := fn.Body[len(fn.Body)-1]
	if last.Op != OpConstInt || last.Imm != 6 {
		t.Errorf("final inst = %+v want ConstInt 6", last)
	}
	// Running pipeline again should not change anything.
	beforeLen := len(fn.Body)
	for _, p := range pipe {
		p.Run(fn)
	}
	if len(fn.Body) != beforeLen {
		t.Errorf("pipeline not idempotent: %d -> %d", beforeLen, len(fn.Body))
	}
}

func TestRunPassesOverModule(t *testing.T) {
	m := &Module{
		Fns: []*Func{
			{
				Body:   []Inst{cint(1, 5), cint(2, 6), bin(3, OpAdd, 1, 2)},
				Result: Value{ID: 3, Type: types.I32()},
			},
			{
				Body:   []Inst{cint(1, 0), cint(2, 0), bin(3, OpMul, 1, 2)},
				Result: Value{ID: 3, Type: types.I32()},
			},
		},
	}
	RunPasses(m, DefaultPipeline())
	for i, fn := range m.Fns {
		last := fn.Body[len(fn.Body)-1]
		if last.Op != OpConstInt {
			t.Errorf("fn[%d] last op = %v", i, last.Op)
		}
	}
	if v := m.Fns[0].Body[len(m.Fns[0].Body)-1].Imm; v != 11 {
		t.Errorf("fn[0] = %d want 11", v)
	}
	if v := m.Fns[1].Body[len(m.Fns[1].Body)-1].Imm; v != 0 {
		t.Errorf("fn[1] = %d want 0", v)
	}
}
