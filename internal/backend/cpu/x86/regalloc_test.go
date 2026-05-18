package x86

import (
	"testing"

	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/types"
)

// TestRegallocCoalesceCopyXMM verifies that an OpCopy whose source dies
// at the copy instruction shares a register with the destination. The
// post-coalesce isel will then elide the `movss` because dst == src.
func TestRegallocCoalesceCopyXMM(t *testing.T) {
	// Build a small MIR func:
	//   v0 = OpConstFloat 1.0
	//   v1 = OpCopy v0       // v0 dies here
	//   ret v1
	f32 := types.F32()
	fn := &mir.Func{
		Name:    "k",
		RetType: f32,
		Body: []mir.Inst{
			{Result: mir.Value{ID: 0, Type: f32}, Op: mir.OpConstFloat, ImmF: 1.0},
			{Result: mir.Value{ID: 1, Type: f32}, Op: mir.OpCopy, Args: []mir.Value{{ID: 0, Type: f32}}},
		},
		Result: mir.Value{ID: 1, Type: f32},
	}

	a := Allocate(fn)
	x0, ok0 := a.GetXMM(0)
	x1, ok1 := a.GetXMM(1)
	if !ok0 || !ok1 {
		t.Fatalf("expected XMM allocation for both v0 and v1; got ok0=%v ok1=%v", ok0, ok1)
	}
	if x0 != x1 {
		t.Errorf("coalescing failed: v0 -> %v, v1 -> %v; want equal", x0, x1)
	}
}

// TestRegallocNoCoalesceWhenSrcStillLive verifies that we do NOT
// coalesce when the source's live range extends past the copy (it's
// used again later). Coalescing in that case would clobber the source.
func TestRegallocNoCoalesceWhenSrcStillLive(t *testing.T) {
	// v0 = OpConstFloat 1.0
	// v1 = OpCopy v0
	// v2 = OpAddF32 v0, v1   // v0 still live at instr 2, after the copy
	// ret v2
	f32 := types.F32()
	fn := &mir.Func{
		Name:    "k",
		RetType: f32,
		Body: []mir.Inst{
			{Result: mir.Value{ID: 0, Type: f32}, Op: mir.OpConstFloat, ImmF: 1.0},
			{Result: mir.Value{ID: 1, Type: f32}, Op: mir.OpCopy, Args: []mir.Value{{ID: 0, Type: f32}}},
			{Result: mir.Value{ID: 2, Type: f32}, Op: mir.OpAddF32, Args: []mir.Value{{ID: 0, Type: f32}, {ID: 1, Type: f32}}},
		},
		Result: mir.Value{ID: 2, Type: f32},
	}

	a := Allocate(fn)
	x0, ok0 := a.GetXMM(0)
	x1, ok1 := a.GetXMM(1)
	if !ok0 || !ok1 {
		t.Fatalf("expected XMM allocation for both v0 and v1; got ok0=%v ok1=%v", ok0, ok1)
	}
	if x0 == x1 {
		t.Errorf("incorrect coalescing: v0 and v1 share %v but v0 is still live after the copy", x0)
	}
}

// TestRegallocCoalesceCopyGPR verifies coalescing on the GPR path
// (integer copy, source dies).
func TestRegallocCoalesceCopyGPR(t *testing.T) {
	// v0 = OpConstInt 7
	// v1 = OpCopy v0
	// ret v1
	i32 := types.I32()
	fn := &mir.Func{
		Name:    "k",
		RetType: i32,
		Body: []mir.Inst{
			{Result: mir.Value{ID: 0, Type: i32}, Op: mir.OpConstInt, Imm: 7},
			{Result: mir.Value{ID: 1, Type: i32}, Op: mir.OpCopy, Args: []mir.Value{{ID: 0, Type: i32}}},
		},
		Result: mir.Value{ID: 1, Type: i32},
	}

	a := Allocate(fn)
	r0, ok0 := a.GetReg(0)
	r1, ok1 := a.GetReg(1)
	if !ok0 || !ok1 {
		t.Fatalf("expected GPR allocation for both v0 and v1; got ok0=%v ok1=%v", ok0, ok1)
	}
	if r0 != r1 {
		t.Errorf("coalescing failed: v0 -> %v, v1 -> %v; want equal", r0, r1)
	}
}
