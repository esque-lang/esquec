// Package ptx implements the NVIDIA PTX (Parallel Thread Execution) backend.
// PTX is a stable, portable intermediate representation for NVIDIA GPUs.
//
// v0.8 introduces kernel compilation, device memory model, and CUDA runtime integration.
package ptx

import (
	"fmt"
	"strings"

	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/types"
)

// Module represents a compiled PTX module.
type Module struct {
	Version    string     // PTX ISA version (e.g., "8.0")
	Target     string     // Target GPU architecture (e.g., "sm_75")
	AddressSize int       // Address size in bits (32 or 64)
	Kernels    []*Kernel
	Functions  []*Function
	Globals    []*Global
}

// Kernel represents a GPU kernel function.
type Kernel struct {
	Name       string
	Params     []Param
	Body       []Inst
	RegDecls   []RegDecl // Register declarations
	SharedMem  int64     // Shared memory size in bytes
}

// Function represents a device function (callable from kernels).
type Function struct {
	Name     string
	Params   []Param
	RetType  *types.Type
	Body     []Inst
	RegDecls []RegDecl
}

// Global represents a global memory variable.
type Global struct {
	Name      string
	Type      *types.Type
	Size      int64  // Size in bytes
	Alignment int64
	Init      []byte // Initial value (nil for uninitialized)
}

// Param represents a kernel or function parameter.
type Param struct {
	Name string
	Type *types.Type
	Size int64 // Size in bytes
}

// RegDecl declares a set of registers of a specific type.
type RegDecl struct {
	Type  RegType
	Count int
	Name  string // Base name (e.g., "r" for r0, r1, ...)
}

// RegType represents PTX register types.
type RegType int

const (
	RegPred  RegType = iota // 1-bit predicate
	RegB8                   // 8-bit
	RegB16                  // 16-bit
	RegB32                  // 32-bit
	RegB64                  // 64-bit
	RegF32                  // 32-bit float
	RegF64                  // 64-bit float
)

func (r RegType) String() string {
	switch r {
	case RegPred:
		return ".pred"
	case RegB8:
		return ".b8"
	case RegB16:
		return ".b16"
	case RegB32:
		return ".b32"
	case RegB64:
		return ".b64"
	case RegF32:
		return ".f32"
	case RegF64:
		return ".f64"
	}
	return ".unknown"
}

// Op represents PTX instruction opcodes.
type Op int

const (
	OpInvalid Op = iota

	// Memory operations
	OpLd      // Load from memory
	OpSt      // Store to memory
	OpMov     // Move/copy
	OpCvt     // Convert type

	// Arithmetic (integer)
	OpAdd     // Add
	OpSub     // Subtract
	OpMul     // Multiply (low bits)
	OpMad     // Multiply-add
	OpDiv     // Divide
	OpRem     // Remainder
	OpAbs     // Absolute value
	OpNeg     // Negate
	OpMin     // Minimum
	OpMax     // Maximum

	// Arithmetic (float)
	OpFAdd    // Float add
	OpFSub    // Float subtract
	OpFMul    // Float multiply
	OpFMa     // Fused multiply-add
	OpFDiv    // Float divide
	OpFRcp    // Reciprocal
	OpFSqrt   // Square root
	OpFRsqrt  // Reciprocal square root
	OpFSin    // Sine
	OpFCos    // Cosine
	OpFExp2   // 2^x
	OpFLg2    // log2(x)
	OpFAbs    // Float absolute value
	OpFNeg    // Float negate
	OpFMin    // Float minimum
	OpFMax    // Float maximum

	// Comparison and selection
	OpSetp    // Set predicate based on comparison
	OpSelp    // Select based on predicate
	OpSlct    // Select (non-predicate)

	// Logical/bitwise
	OpAnd     // Bitwise AND
	OpOr      // Bitwise OR
	OpXor     // Bitwise XOR
	OpNot     // Bitwise NOT
	OpShl     // Shift left
	OpShr     // Shift right

	// Control flow
	OpBra     // Branch
	OpCall    // Call function
	OpRet     // Return
	OpExit    // Exit kernel

	// Synchronization
	OpBar     // Barrier synchronization
	OpMembar  // Memory barrier
	OpAtom    // Atomic operation

	// Special
	OpTid     // Thread ID
	OpNtid    // Number of threads
	OpCtaid   // CTA (block) ID
	OpNctaid  // Number of CTAs
	OpLaneid  // Lane ID within warp
	OpWarpid  // Warp ID within CTA

	// Labels (pseudo-op)
	OpLabel   // Label marker
)

// StateSpace represents PTX memory state spaces.
type StateSpace int

const (
	SSGeneric StateSpace = iota // Generic (determined at runtime)
	SSGlobal                    // Global memory
	SSShared                    // Shared memory (per-block)
	SSLocal                     // Local memory (per-thread stack)
	SSConst                     // Constant memory (read-only)
	SSParam                     // Parameter memory
	SSReg                       // Registers
)

func (s StateSpace) String() string {
	switch s {
	case SSGeneric:
		return ""
	case SSGlobal:
		return ".global"
	case SSShared:
		return ".shared"
	case SSLocal:
		return ".local"
	case SSConst:
		return ".const"
	case SSParam:
		return ".param"
	case SSReg:
		return ".reg"
	}
	return ".unknown"
}

// Inst represents a PTX instruction.
type Inst struct {
	Op         Op
	Type       RegType     // Data type for the operation
	StateSpace StateSpace  // Memory state space (for load/store)
	Dest       Operand     // Destination operand
	Src        []Operand   // Source operands
	Pred       string      // Predicate register (for conditional exec)
	PredNeg    bool        // Negate predicate
	Label      string      // For branches and labels
	CmpOp      CmpOp       // Comparison operator (for setp)
}

// Operand represents an instruction operand.
type Operand struct {
	Kind    OperandKind
	Reg     string  // Register name
	Imm     int64   // Immediate value
	ImmF    float64 // Float immediate
	Addr    string  // Address label or register
	Offset  int64   // Address offset
}

// OperandKind represents the kind of operand.
type OperandKind int

const (
	OpndReg    OperandKind = iota // Register
	OpndImm                       // Immediate integer
	OpndImmF                      // Immediate float
	OpndAddr                      // Memory address [reg+offset]
	OpndLabel                     // Label reference
	OpndParam                     // Parameter reference
)

// CmpOp represents comparison operators for setp.
type CmpOp int

const (
	CmpEq  CmpOp = iota // Equal
	CmpNe               // Not equal
	CmpLt               // Less than
	CmpLe               // Less than or equal
	CmpGt               // Greater than
	CmpGe               // Greater than or equal
	// Float-specific
	CmpLtu              // Less than (unordered)
	CmpLeu              // Less than or equal (unordered)
	CmpGtu              // Greater than (unordered)
	CmpGeu              // Greater than or equal (unordered)
	CmpNum              // Ordered (neither is NaN)
	CmpNan              // Unordered (at least one is NaN)
)

func (c CmpOp) String() string {
	switch c {
	case CmpEq:
		return "eq"
	case CmpNe:
		return "ne"
	case CmpLt:
		return "lt"
	case CmpLe:
		return "le"
	case CmpGt:
		return "gt"
	case CmpGe:
		return "ge"
	case CmpLtu:
		return "ltu"
	case CmpLeu:
		return "leu"
	case CmpGtu:
		return "gtu"
	case CmpGeu:
		return "geu"
	case CmpNum:
		return "num"
	case CmpNan:
		return "nan"
	}
	return "?"
}

// Compiler compiles MIR to PTX.
type Compiler struct {
	mod       *Module
	regCount  map[RegType]int
	labelCount int
}

// NewCompiler creates a new PTX compiler.
func NewCompiler() *Compiler {
	return &Compiler{
		mod: &Module{
			Version:     "8.0",
			Target:      "sm_75", // Default to Turing architecture
			AddressSize: 64,
		},
		regCount: make(map[RegType]int),
	}
}

// CompileFunction compiles a MIR function as a device function (callable from kernels).
func (c *Compiler) CompileFunction(f *mir.Func) (*Function, error) {
	fn := &Function{
		Name:    f.Name,
		RetType: f.RetType,
	}

	// Convert parameters
	for _, p := range f.Params {
		fn.Params = append(fn.Params, Param{
			Name: p.Name,
			Type: p.Type,
			Size: typeSize(p.Type),
		})
	}

	// Reset register counts for this function
	c.regCount = make(map[RegType]int)
	c.labelCount = 0

	// Lower MIR instructions to PTX
	for _, in := range f.Body {
		insts, err := c.lowerInst(in)
		if err != nil {
			return nil, fmt.Errorf("compiling %s: %w", f.Name, err)
		}
		fn.Body = append(fn.Body, insts...)
	}

	// Add return instruction
	fn.Body = append(fn.Body, Inst{Op: OpRet})

	// Generate register declarations
	for rt, count := range c.regCount {
		if count > 0 {
			fn.RegDecls = append(fn.RegDecls, RegDecl{
				Type:  rt,
				Count: count,
				Name:  regPrefix(rt),
			})
		}
	}

	return fn, nil
}

// CompileKernel compiles a MIR function as a GPU kernel.
func (c *Compiler) CompileKernel(f *mir.Func) (*Kernel, error) {
	k := &Kernel{
		Name: f.Name,
	}

	// Convert parameters
	for _, p := range f.Params {
		k.Params = append(k.Params, Param{
			Name: p.Name,
			Type: p.Type,
			Size: typeSize(p.Type),
		})
	}

	// Reset register counts for this kernel
	c.regCount = make(map[RegType]int)
	c.labelCount = 0

	// Lower MIR instructions to PTX
	for _, in := range f.Body {
		insts, err := c.lowerInst(in)
		if err != nil {
			return nil, fmt.Errorf("compiling %s: %w", f.Name, err)
		}
		k.Body = append(k.Body, insts...)
	}

	// Add exit instruction
	k.Body = append(k.Body, Inst{Op: OpExit})

	// Generate register declarations
	for rt, count := range c.regCount {
		if count > 0 {
			k.RegDecls = append(k.RegDecls, RegDecl{
				Type:  rt,
				Count: count,
				Name:  regPrefix(rt),
			})
		}
	}

	return k, nil
}

// lowerInst converts a MIR instruction to PTX instructions.
func (c *Compiler) lowerInst(in mir.Inst) ([]Inst, error) {
	var out []Inst

	switch in.Op {
	case mir.OpConstInt:
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegB64,
			Dest: c.reg(in.Result.ID, RegB64),
			Src:  []Operand{{Kind: OpndImm, Imm: in.Imm}},
		})

	case mir.OpConstFloat:
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{{Kind: OpndImmF, ImmF: in.ImmF}},
		})

	case mir.OpAdd:
		if isIntType(in.Result.Type) {
			out = append(out, Inst{
				Op:   OpAdd,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		} else {
			out = append(out, Inst{
				Op:   OpFAdd,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		}

	case mir.OpSub:
		if isIntType(in.Result.Type) {
			out = append(out, Inst{
				Op:   OpSub,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		} else {
			out = append(out, Inst{
				Op:   OpFSub,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		}

	case mir.OpMul:
		if isIntType(in.Result.Type) {
			out = append(out, Inst{
				Op:   OpMul,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		} else {
			out = append(out, Inst{
				Op:   OpFMul,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		}

	case mir.OpDiv:
		if isIntType(in.Result.Type) {
			out = append(out, Inst{
				Op:   OpDiv,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		} else {
			out = append(out, Inst{
				Op:   OpFDiv,
				Type: regTypeFor(in.Result.Type),
				Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
				Src: []Operand{
					c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
					c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
				},
			})
		}

	case mir.OpLoad32:
		out = append(out, Inst{
			Op:         OpLd,
			Type:       RegF32,
			StateSpace: SSGlobal,
			Dest:       c.reg(in.Result.ID, RegF32),
			Src: []Operand{{
				Kind:   OpndAddr,
				Reg:    c.regName(in.Args[0].ID, RegB64),
				Offset: in.Imm,
			}},
		})

	case mir.OpStore32:
		out = append(out, Inst{
			Op:         OpSt,
			Type:       RegF32,
			StateSpace: SSGlobal,
			Dest: Operand{
				Kind:   OpndAddr,
				Reg:    c.regName(in.Args[0].ID, RegB64),
				Offset: in.Imm,
			},
			Src: []Operand{c.reg(in.Args[1].ID, RegF32)},
		})

	case mir.OpAddF32:
		out = append(out, Inst{
			Op:   OpFAdd,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src: []Operand{
				c.reg(in.Args[0].ID, RegF32),
				c.reg(in.Args[1].ID, RegF32),
			},
		})

	case mir.OpMulF32:
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src: []Operand{
				c.reg(in.Args[0].ID, RegF32),
				c.reg(in.Args[1].ID, RegF32),
			},
		})

	case mir.OpCastF32ToI32:
		out = append(out, Inst{
			Op:   OpCvt,
			Type: RegB32,
			Dest: c.reg(in.Result.ID, RegB32),
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})

	case mir.OpCastI32ToF32:
		out = append(out, Inst{
			Op:   OpCvt,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{c.reg(in.Args[0].ID, RegB32)},
		})

	// Control flow operations (v0.9)
	case mir.OpLabel:
		out = append(out, Inst{
			Op:    OpLabel,
			Label: fmt.Sprintf("L%d", in.LabelID),
		})

	case mir.OpCmpLt:
		// setp.lt.s32 p0, src1, src2
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpLt,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpJumpIf:
		// @p0 bra L0
		out = append(out, Inst{
			Op:    OpBra,
			Pred:  c.regName(in.Args[0].ID, RegPred),
			Label: fmt.Sprintf("L%d", in.LabelID),
		})

	case mir.OpJump:
		// bra L0
		out = append(out, Inst{
			Op:    OpBra,
			Label: fmt.Sprintf("L%d", in.LabelID),
		})

	case mir.OpEq:
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpEq,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpNe:
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpNe,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpLt:
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpLt,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpLe:
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpLe,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpGt:
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpGt,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpGe:
		out = append(out, Inst{
			Op:    OpSetp,
			Type:  regTypeFor(in.Args[0].Type),
			CmpOp: CmpGe,
			Dest:  c.reg(in.Result.ID, RegPred),
			Src: []Operand{
				c.reg(in.Args[0].ID, regTypeFor(in.Args[0].Type)),
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)),
			},
		})

	case mir.OpSelect:
		// selp.type dest, src1, src2, pred
		out = append(out, Inst{
			Op:   OpSelp,
			Type: regTypeFor(in.Result.Type),
			Dest: c.reg(in.Result.ID, regTypeFor(in.Result.Type)),
			Src: []Operand{
				c.reg(in.Args[1].ID, regTypeFor(in.Args[1].Type)), // true value
				c.reg(in.Args[2].ID, regTypeFor(in.Args[2].Type)), // false value
				c.reg(in.Args[0].ID, RegPred),                     // predicate
			},
		})

	case mir.OpCopy:
		rt := regTypeFor(in.Result.Type)
		out = append(out, Inst{
			Op:   OpMov,
			Type: rt,
			Dest: c.reg(in.Result.ID, rt),
			Src:  []Operand{c.reg(in.Args[0].ID, rt)},
		})

	case mir.OpCall:
		// Device function call: call funcname, (args), (retval)
		callInst := Inst{
			Op:    OpCall,
			Label: in.FnName,
		}
		// Add arguments
		for _, arg := range in.Args {
			callInst.Src = append(callInst.Src, c.reg(arg.ID, regTypeFor(arg.Type)))
		}
		// Set destination if there's a return value
		if in.Result.ID > 0 {
			callInst.Type = regTypeFor(in.Result.Type)
			callInst.Dest = c.reg(in.Result.ID, regTypeFor(in.Result.Type))
		}
		out = append(out, callInst)

	// Math operations (v0.9 - neural network support)
	case mir.OpExpF32:
		// exp2(x * log2(e)) = exp(x)
		// For now, use approx ex2 with conversion
		logE := c.freshReg(RegF32)
		scaled := c.freshReg(RegF32)
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: logE},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 1.4426950408889634}}, // log2(e)
		})
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: scaled},
			Src: []Operand{
				c.reg(in.Args[0].ID, RegF32),
				{Kind: OpndReg, Reg: logE},
			},
		})
		out = append(out, Inst{
			Op:   OpFExp2,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{{Kind: OpndReg, Reg: scaled}},
		})

	case mir.OpLogF32:
		// log(x) = log2(x) / log2(e)
		log2Val := c.freshReg(RegF32)
		invLogE := c.freshReg(RegF32)
		out = append(out, Inst{
			Op:   OpFLg2,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: log2Val},
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: invLogE},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 0.6931471805599453}}, // 1/log2(e) = ln(2)
		})
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src: []Operand{
				{Kind: OpndReg, Reg: log2Val},
				{Kind: OpndReg, Reg: invLogE},
			},
		})

	case mir.OpSqrtF32:
		out = append(out, Inst{
			Op:   OpFSqrt,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})

	case mir.OpSinF32:
		out = append(out, Inst{
			Op:   OpFSin,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})

	case mir.OpCosF32:
		out = append(out, Inst{
			Op:   OpFCos,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})

	case mir.OpAbsF32:
		out = append(out, Inst{
			Op:   OpFAbs,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})

	case mir.OpMinF32:
		out = append(out, Inst{
			Op:   OpFMin,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src: []Operand{
				c.reg(in.Args[0].ID, RegF32),
				c.reg(in.Args[1].ID, RegF32),
			},
		})

	case mir.OpMaxF32:
		out = append(out, Inst{
			Op:   OpFMax,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src: []Operand{
				c.reg(in.Args[0].ID, RegF32),
				c.reg(in.Args[1].ID, RegF32),
			},
		})

	case mir.OpReLUF32:
		// relu(x) = max(0, x)
		zero := c.freshReg(RegF32)
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: zero},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 0.0}},
		})
		out = append(out, Inst{
			Op:   OpFMax,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src: []Operand{
				{Kind: OpndReg, Reg: zero},
				c.reg(in.Args[0].ID, RegF32),
			},
		})

	case mir.OpSigmoidF32:
		// sigmoid(x) = 1 / (1 + exp(-x))
		negX := c.freshReg(RegF32)
		expNegX := c.freshReg(RegF32)
		onePlusExp := c.freshReg(RegF32)
		one := c.freshReg(RegF32)
		logE := c.freshReg(RegF32)
		scaled := c.freshReg(RegF32)

		// -x
		out = append(out, Inst{
			Op:   OpFNeg,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: negX},
			Src:  []Operand{c.reg(in.Args[0].ID, RegF32)},
		})
		// exp(-x) via ex2
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: logE},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 1.4426950408889634}},
		})
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: scaled},
			Src:  []Operand{{Kind: OpndReg, Reg: negX}, {Kind: OpndReg, Reg: logE}},
		})
		out = append(out, Inst{
			Op:   OpFExp2,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: expNegX},
			Src:  []Operand{{Kind: OpndReg, Reg: scaled}},
		})
		// 1 + exp(-x)
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: one},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 1.0}},
		})
		out = append(out, Inst{
			Op:   OpFAdd,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: onePlusExp},
			Src:  []Operand{{Kind: OpndReg, Reg: one}, {Kind: OpndReg, Reg: expNegX}},
		})
		// 1 / (1 + exp(-x))
		out = append(out, Inst{
			Op:   OpFRcp,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{{Kind: OpndReg, Reg: onePlusExp}},
		})

	case mir.OpTanhF32:
		// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
		// Or: tanh(x) = 2*sigmoid(2x) - 1
		two := c.freshReg(RegF32)
		twoX := c.freshReg(RegF32)
		exp2X := c.freshReg(RegF32)
		one := c.freshReg(RegF32)
		num := c.freshReg(RegF32)
		denom := c.freshReg(RegF32)
		rcpDenom := c.freshReg(RegF32)
		logE := c.freshReg(RegF32)
		scaled := c.freshReg(RegF32)

		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: two},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 2.0}},
		})
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: twoX},
			Src:  []Operand{{Kind: OpndReg, Reg: two}, c.reg(in.Args[0].ID, RegF32)},
		})
		// exp(2x)
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: logE},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 1.4426950408889634}},
		})
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: scaled},
			Src:  []Operand{{Kind: OpndReg, Reg: twoX}, {Kind: OpndReg, Reg: logE}},
		})
		out = append(out, Inst{
			Op:   OpFExp2,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: exp2X},
			Src:  []Operand{{Kind: OpndReg, Reg: scaled}},
		})
		// exp(2x) - 1
		out = append(out, Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: one},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 1.0}},
		})
		out = append(out, Inst{
			Op:   OpFSub,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: num},
			Src:  []Operand{{Kind: OpndReg, Reg: exp2X}, {Kind: OpndReg, Reg: one}},
		})
		// exp(2x) + 1
		out = append(out, Inst{
			Op:   OpFAdd,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: denom},
			Src:  []Operand{{Kind: OpndReg, Reg: exp2X}, {Kind: OpndReg, Reg: one}},
		})
		// result
		out = append(out, Inst{
			Op:   OpFRcp,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: rcpDenom},
			Src:  []Operand{{Kind: OpndReg, Reg: denom}},
		})
		out = append(out, Inst{
			Op:   OpFMul,
			Type: RegF32,
			Dest: c.reg(in.Result.ID, RegF32),
			Src:  []Operand{{Kind: OpndReg, Reg: num}, {Kind: OpndReg, Reg: rcpDenom}},
		})

	default:
		// Skip unsupported ops for now
		return nil, nil
	}

	return out, nil
}

func (c *Compiler) reg(id int, rt RegType) Operand {
	return Operand{
		Kind: OpndReg,
		Reg:  c.regName(id, rt),
	}
}

func (c *Compiler) regName(id int, rt RegType) string {
	// Track register usage
	if id >= c.regCount[rt] {
		c.regCount[rt] = id + 1
	}
	return fmt.Sprintf("%s%d", regPrefix(rt), id)
}

func (c *Compiler) freshLabel() string {
	l := fmt.Sprintf("L%d", c.labelCount)
	c.labelCount++
	return l
}

// freshReg creates a fresh temporary register name.
func (c *Compiler) freshReg(rt RegType) string {
	id := c.regCount[rt]
	c.regCount[rt] = id + 1
	return fmt.Sprintf("%s%d", regPrefix(rt), id)
}

func regPrefix(rt RegType) string {
	switch rt {
	case RegPred:
		return "p"
	case RegB8:
		return "rb"
	case RegB16:
		return "rh"
	case RegB32:
		return "r"
	case RegB64:
		return "rd"
	case RegF32:
		return "f"
	case RegF64:
		return "fd"
	}
	return "?"
}

func regTypeFor(t *types.Type) RegType {
	if t == nil {
		return RegB64
	}
	switch t.K {
	case types.KBool:
		return RegPred
	case types.KI8, types.KU8:
		return RegB8
	case types.KI16, types.KU16:
		return RegB16
	case types.KI32, types.KU32:
		return RegB32
	case types.KI64, types.KU64:
		return RegB64
	case types.KF32:
		return RegF32
	case types.KF64:
		return RegF64
	}
	return RegB64
}

func isIntType(t *types.Type) bool {
	if t == nil {
		return true
	}
	switch t.K {
	case types.KI8, types.KI16, types.KI32, types.KI64,
		types.KU8, types.KU16, types.KU32, types.KU64:
		return true
	}
	return false
}

func typeSize(t *types.Type) int64 {
	if t == nil {
		return 8
	}
	switch t.K {
	case types.KBool, types.KI8, types.KU8:
		return 1
	case types.KI16, types.KU16:
		return 2
	case types.KI32, types.KU32, types.KF32:
		return 4
	case types.KI64, types.KU64, types.KF64:
		return 8
	}
	// For tensors, return pointer size
	if t.IsTensor() {
		return 8
	}
	return 8
}

// Emit generates PTX assembly text for the module.
func (m *Module) Emit() string {
	var b strings.Builder

	// Header
	fmt.Fprintf(&b, ".version %s\n", m.Version)
	fmt.Fprintf(&b, ".target %s\n", m.Target)
	fmt.Fprintf(&b, ".address_size %d\n\n", m.AddressSize)

	// Global declarations
	for _, g := range m.Globals {
		fmt.Fprintf(&b, ".global .align %d .b8 %s[%d];\n",
			g.Alignment, g.Name, g.Size)
	}
	if len(m.Globals) > 0 {
		b.WriteString("\n")
	}

	// Kernels
	for _, k := range m.Kernels {
		k.emit(&b)
		b.WriteString("\n")
	}

	// Device functions
	for _, f := range m.Functions {
		f.emit(&b)
		b.WriteString("\n")
	}

	return b.String()
}

func (k *Kernel) emit(b *strings.Builder) {
	// Entry point declaration
	fmt.Fprintf(b, ".visible .entry %s(\n", k.Name)

	// Parameters
	for i, p := range k.Params {
		comma := ","
		if i == len(k.Params)-1 {
			comma = ""
		}
		fmt.Fprintf(b, "    .param .u64 %s%s\n", p.Name, comma)
	}
	b.WriteString(")\n{\n")

	// Register declarations
	for _, rd := range k.RegDecls {
		fmt.Fprintf(b, "    .reg %s %s<%d>;\n", rd.Type, rd.Name, rd.Count)
	}
	b.WriteString("\n")

	// Load parameters
	for _, p := range k.Params {
		fmt.Fprintf(b, "    ld.param.u64 rd_%s, [%s];\n", p.Name, p.Name)
	}
	b.WriteString("\n")

	// Instructions
	for _, in := range k.Body {
		b.WriteString("    ")
		in.emit(b)
		b.WriteString("\n")
	}

	b.WriteString("}\n")
}

func (f *Function) emit(b *strings.Builder) {
	// Function declaration with optional return value
	if f.RetType != nil && f.RetType.K != 0 {
		rt := regTypeFor(f.RetType)
		fmt.Fprintf(b, ".func (%s retval) %s(\n", rt, f.Name)
	} else {
		fmt.Fprintf(b, ".func %s(\n", f.Name)
	}

	for i, p := range f.Params {
		comma := ","
		if i == len(f.Params)-1 {
			comma = ""
		}
		pt := regTypeFor(p.Type)
		fmt.Fprintf(b, "    .param %s %s%s\n", pt, p.Name, comma)
	}
	b.WriteString(")\n{\n")

	for _, rd := range f.RegDecls {
		fmt.Fprintf(b, "    .reg %s %s<%d>;\n", rd.Type, rd.Name, rd.Count)
	}
	b.WriteString("\n")

	// Load parameters into registers
	for _, p := range f.Params {
		pt := regTypeFor(p.Type)
		fmt.Fprintf(b, "    ld.param%s %s_%s, [%s];\n", pt, regPrefix(pt), p.Name, p.Name)
	}
	if len(f.Params) > 0 {
		b.WriteString("\n")
	}

	for _, in := range f.Body {
		b.WriteString("    ")
		in.emit(b)
		b.WriteString("\n")
	}

	b.WriteString("}\n")
}

func (in *Inst) emit(b *strings.Builder) {
	// Predicate
	if in.Pred != "" {
		if in.PredNeg {
			fmt.Fprintf(b, "@!%s ", in.Pred)
		} else {
			fmt.Fprintf(b, "@%s ", in.Pred)
		}
	}

	switch in.Op {
	case OpMov:
		fmt.Fprintf(b, "mov%s %s, ", in.Type, in.Dest.String())
		b.WriteString(in.Src[0].String())

	case OpLd:
		fmt.Fprintf(b, "ld%s%s %s, [%s]",
			in.StateSpace, in.Type, in.Dest.String(),
			formatAddr(in.Src[0]))

	case OpSt:
		fmt.Fprintf(b, "st%s%s [%s], %s",
			in.StateSpace, in.Type, formatAddr(in.Dest),
			in.Src[0].String())

	case OpAdd:
		fmt.Fprintf(b, "add%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpSub:
		fmt.Fprintf(b, "sub%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpMul:
		fmt.Fprintf(b, "mul.lo%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpDiv:
		fmt.Fprintf(b, "div%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFAdd:
		fmt.Fprintf(b, "add%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFSub:
		fmt.Fprintf(b, "sub%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFMul:
		fmt.Fprintf(b, "mul%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFDiv:
		fmt.Fprintf(b, "div.approx%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFMa:
		fmt.Fprintf(b, "fma.rn%s %s, %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String(), in.Src[2].String())

	case OpCvt:
		fmt.Fprintf(b, "cvt.rzi%s%s %s, %s",
			in.Type, srcType(in.Src[0]), in.Dest.String(),
			in.Src[0].String())

	case OpSetp:
		fmt.Fprintf(b, "setp.%s%s %s, %s, %s",
			in.CmpOp, in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpCall:
		// Device function call
		if in.Dest.Kind == OpndReg {
			fmt.Fprintf(b, "call (%s), %s, (", in.Dest.String(), in.Label)
		} else {
			fmt.Fprintf(b, "call %s, (", in.Label)
		}
		for i, src := range in.Src {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(src.String())
		}
		b.WriteString(")")

	case OpFSqrt:
		fmt.Fprintf(b, "sqrt.approx%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFSin:
		fmt.Fprintf(b, "sin.approx%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFCos:
		fmt.Fprintf(b, "cos.approx%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFAbs:
		fmt.Fprintf(b, "abs%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFNeg:
		fmt.Fprintf(b, "neg%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFMin:
		fmt.Fprintf(b, "min%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFMax:
		fmt.Fprintf(b, "max%s %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String())

	case OpFExp2:
		fmt.Fprintf(b, "ex2.approx%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFLg2:
		fmt.Fprintf(b, "lg2.approx%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpFRcp:
		fmt.Fprintf(b, "rcp.approx%s %s, %s",
			in.Type, in.Dest.String(), in.Src[0].String())

	case OpBra:
		fmt.Fprintf(b, "bra %s", in.Label)

	case OpRet:
		b.WriteString("ret")

	case OpExit:
		b.WriteString("exit")

	case OpBar:
		b.WriteString("bar.sync 0")

	case OpSelp:
		fmt.Fprintf(b, "selp%s %s, %s, %s, %s",
			in.Type, in.Dest.String(),
			in.Src[0].String(), in.Src[1].String(), in.Src[2].String())

	case OpLabel:
		// Labels don't have semicolons
		fmt.Fprintf(b, "%s:", in.Label)
		return

	default:
		fmt.Fprintf(b, "// unsupported op %d", in.Op)
	}

	b.WriteString(";")
}

func (o Operand) String() string {
	switch o.Kind {
	case OpndReg:
		return o.Reg
	case OpndImm:
		return fmt.Sprintf("%d", o.Imm)
	case OpndImmF:
		return fmt.Sprintf("0F%08X", floatBits(o.ImmF))
	case OpndAddr:
		if o.Offset != 0 {
			return fmt.Sprintf("%s+%d", o.Reg, o.Offset)
		}
		return o.Reg
	case OpndLabel:
		return o.Addr
	case OpndParam:
		return fmt.Sprintf("[%s]", o.Addr)
	}
	return "?"
}

func formatAddr(o Operand) string {
	if o.Offset != 0 {
		return fmt.Sprintf("%s+%d", o.Reg, o.Offset)
	}
	return o.Reg
}

func srcType(o Operand) string {
	// Infer source type from register name prefix
	if len(o.Reg) > 0 {
		switch o.Reg[0] {
		case 'f':
			return ".f32"
		case 'r':
			if len(o.Reg) > 1 {
				switch o.Reg[1] {
				case 'd':
					return ".s64"
				case 'h':
					return ".s16"
				case 'b':
					return ".s8"
				}
			}
			return ".s32"
		case 'p':
			return ".pred"
		}
	}
	return ".s32"
}

func floatBits(f float64) uint32 {
	return uint32(f * (1 << 23)) // Simplified; real impl would use math.Float32bits
}
