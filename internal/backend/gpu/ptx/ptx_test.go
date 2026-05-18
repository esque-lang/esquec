package ptx

import (
	"strings"
	"testing"

	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/types"
)

func TestCompileSimpleKernel(t *testing.T) {
	// Create a simple MIR function that adds two floats
	fn := &mir.Func{
		Name: "add_kernel",
		Params: []mir.Param{
			{Name: "a", Type: types.F32()},
			{Name: "b", Type: types.F32()},
		},
		RetType: types.F32(),
		Body: []mir.Inst{
			{
				Result: mir.Value{ID: 0, Type: types.F32()},
				Op:     mir.OpConstFloat,
				ImmF:   1.0,
			},
			{
				Result: mir.Value{ID: 1, Type: types.F32()},
				Op:     mir.OpConstFloat,
				ImmF:   2.0,
			},
			{
				Result: mir.Value{ID: 2, Type: types.F32()},
				Op:     mir.OpAddF32,
				Args:   []mir.Value{{ID: 0, Type: types.F32()}, {ID: 1, Type: types.F32()}},
			},
		},
		Result: mir.Value{ID: 2, Type: types.F32()},
	}

	c := NewCompiler()
	k, err := c.CompileKernel(fn)
	if err != nil {
		t.Fatalf("CompileKernel failed: %v", err)
	}

	if k.Name != "add_kernel" {
		t.Errorf("expected kernel name 'add_kernel', got %q", k.Name)
	}

	// Check that we have some instructions
	if len(k.Body) == 0 {
		t.Error("expected non-empty kernel body")
	}

	// Should have an exit instruction at the end
	lastInst := k.Body[len(k.Body)-1]
	if lastInst.Op != OpExit {
		t.Errorf("expected OpExit at end, got %v", lastInst.Op)
	}
}

func TestPTXModuleEmit(t *testing.T) {
	mod := &Module{
		Version:     "8.0",
		Target:      "sm_75",
		AddressSize: 64,
		Kernels: []*Kernel{
			{
				Name: "vector_add",
				Params: []Param{
					{Name: "a", Type: types.F32(), Size: 8},
					{Name: "b", Type: types.F32(), Size: 8},
					{Name: "c", Type: types.F32(), Size: 8},
				},
				RegDecls: []RegDecl{
					{Type: RegB64, Count: 5, Name: "rd"},
					{Type: RegF32, Count: 3, Name: "f"},
				},
				Body: []Inst{
					{
						Op:   OpMov,
						Type: RegF32,
						Dest: Operand{Kind: OpndReg, Reg: "f0"},
						Src:  []Operand{{Kind: OpndImmF, ImmF: 1.0}},
					},
					{
						Op:   OpMov,
						Type: RegF32,
						Dest: Operand{Kind: OpndReg, Reg: "f1"},
						Src:  []Operand{{Kind: OpndImmF, ImmF: 2.0}},
					},
					{
						Op:   OpFAdd,
						Type: RegF32,
						Dest: Operand{Kind: OpndReg, Reg: "f2"},
						Src: []Operand{
							{Kind: OpndReg, Reg: "f0"},
							{Kind: OpndReg, Reg: "f1"},
						},
					},
					{
						Op: OpExit,
					},
				},
			},
		},
	}

	ptx := mod.Emit()

	// Check header
	if !strings.Contains(ptx, ".version 8.0") {
		t.Error("expected PTX version header")
	}
	if !strings.Contains(ptx, ".target sm_75") {
		t.Error("expected target declaration")
	}
	if !strings.Contains(ptx, ".address_size 64") {
		t.Error("expected address size")
	}

	// Check kernel entry
	if !strings.Contains(ptx, ".visible .entry vector_add") {
		t.Error("expected kernel entry declaration")
	}

	// Check register declarations
	if !strings.Contains(ptx, ".reg .b64 rd<5>") {
		t.Error("expected b64 register declaration")
	}
	if !strings.Contains(ptx, ".reg .f32 f<3>") {
		t.Error("expected f32 register declaration")
	}

	// Check instructions
	if !strings.Contains(ptx, "add.f32") {
		t.Error("expected add.f32 instruction")
	}
	if !strings.Contains(ptx, "exit;") {
		t.Error("expected exit instruction")
	}

	t.Logf("Generated PTX:\n%s", ptx)
}

func TestRegTypeString(t *testing.T) {
	tests := []struct {
		rt   RegType
		want string
	}{
		{RegPred, ".pred"},
		{RegB8, ".b8"},
		{RegB16, ".b16"},
		{RegB32, ".b32"},
		{RegB64, ".b64"},
		{RegF32, ".f32"},
		{RegF64, ".f64"},
	}

	for _, tt := range tests {
		if got := tt.rt.String(); got != tt.want {
			t.Errorf("RegType(%d).String() = %q, want %q", tt.rt, got, tt.want)
		}
	}
}

func TestStateSpaceString(t *testing.T) {
	tests := []struct {
		ss   StateSpace
		want string
	}{
		{SSGeneric, ""},
		{SSGlobal, ".global"},
		{SSShared, ".shared"},
		{SSLocal, ".local"},
		{SSConst, ".const"},
		{SSParam, ".param"},
	}

	for _, tt := range tests {
		if got := tt.ss.String(); got != tt.want {
			t.Errorf("StateSpace(%d).String() = %q, want %q", tt.ss, got, tt.want)
		}
	}
}

func TestCmpOpString(t *testing.T) {
	tests := []struct {
		op   CmpOp
		want string
	}{
		{CmpEq, "eq"},
		{CmpNe, "ne"},
		{CmpLt, "lt"},
		{CmpLe, "le"},
		{CmpGt, "gt"},
		{CmpGe, "ge"},
	}

	for _, tt := range tests {
		if got := tt.op.String(); got != tt.want {
			t.Errorf("CmpOp(%d).String() = %q, want %q", tt.op, got, tt.want)
		}
	}
}
