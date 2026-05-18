// Package mir is the matrix IR — closer to hardware. For v0.1 it is a
// near-relabeling of CEIR: scalar ops only, no kernels, no transfers.
package mir

import "github.com/esque-lang/esquec/internal/types"

type Module struct {
	Path   string
	Fns    []*Func
	Rodata []RodataEntry
}

// RodataEntry is a constant blob laid out in the .rodata section. Each
// entry gets a local symbol (Sym) that OpRodataPtr instructions resolve
// to via a RIP-relative load.
type RodataEntry struct {
	Sym   string
	Bytes []byte
	Align uint64
}

type Func struct {
	Name    string
	Params  []Param
	RetType *types.Type
	Body    []Inst
	Result  Value
}

type Param struct {
	Name string
	Type *types.Type
}

type Op int

const (
	OpInvalid Op = iota
	OpConstInt
	OpConstFloat
	OpConstBool
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMod
	OpNeg
	OpNot
	OpEq
	OpNe
	OpLt
	OpLe
	OpGt
	OpGe
	OpAnd
	OpOr
	OpCall
	OpSelect
	OpCopy

	// Memory operations for tensors
	OpStackAlloc // Allocate bytes on stack, result is base address (GPR)
	OpRodataPtr  // Load address of a .rodata symbol (FnName=symbol) into a GPR
	OpLoad32     // Load f32 from [base + offset]
	OpStore32    // Store f32 to [base + offset]
	OpLoadI32    // Load i32 from [base + offset] (GPR, zero-extended to 64)
	OpStoreI32   // Store i32 to [base + offset] (GPR)
	OpLoad64     // Load i64 from [base + offset] (GPR)
	OpStore64    // Store i64 to [base + offset] (GPR)
	OpLEA        // Load effective address [base + offset]

	// Loop/control flow operations
	OpLabel   // Label marker (no code, just marks position)
	OpCmpLt   // Compare less than, sets flags
	OpJumpIf  // Conditional jump to label
	OpJump    // Unconditional jump to label

	// Float operations that work on XMM
	OpAddF32
	OpSubF32
	OpMulF32
	OpDivF32

	// Cast operations
	OpCastF32ToI32 // CVTTSS2SI: truncate f32 to i32
	OpCastF64ToI32 // CVTTSD2SI: truncate f64 to i32
	OpCastI32ToF32 // CVTSI2SS: convert i32 to f32
	OpCastI32ToF64 // CVTSI2SD: convert i32 to f64

	// Integer width casts involving i8/u8 (v0.12). Lowered to MOVSX/
	// MOVZX from the source register's low 8 bits into the destination.
	OpCastI8ToI32 // MOVSX r64, r8
	OpCastI8ToI64 // MOVSX r64, r8
	OpCastU8ToI32 // MOVZX r64, r8
	OpCastU8ToI64 // MOVZX r64, r8
	OpCastI32ToI8 // MOVSX r64, r8 (truncate then sign-extend)
	OpCastI32ToU8 // MOVZX r64, r8 (truncate then zero-extend)
	OpCastI64ToI8 // MOVSX r64, r8 (truncate then sign-extend)
	OpCastI64ToU8 // MOVZX r64, r8 (truncate then zero-extend)

	// Matrix operations
	OpMatMul // matrix multiply: Args[0] @ Args[1], stores [M,K,N] in Aux

	// AVX2 packed f32 operations (8 elements per instruction)
	OpLoad256  // Load 8 x f32 from [base + offset]
	OpStore256 // Store 8 x f32 to [base + offset]
	OpAddPS256 // dst = src1 + src2 (8 packed f32)
	OpSubPS256 // dst = src1 - src2 (8 packed f32)
	OpMulPS256 // dst = src1 * src2 (8 packed f32)
	OpDivPS256 // dst = src1 / src2 (8 packed f32)

	// SSE packed f32 operations (4 elements per instruction)
	OpLoad128  // Load 4 x f32 from [base + offset]
	OpStore128 // Store 4 x f32 to [base + offset]
	OpAddPS128 // dst = src1 + src2 (4 packed f32)
	OpSubPS128 // dst = src1 - src2 (4 packed f32)
	OpMulPS128 // dst = src1 * src2 (4 packed f32)
	OpDivPS128 // dst = src1 / src2 (4 packed f32)

	// Horizontal SIMD reduction primitives (v0.10).
	// Building blocks for collapsing a packed f32 vector to a scalar
	// without a scalar accumulator chain.
	OpHAddPS128      // SSE3 haddps: pairwise add of dst's 4 lanes with src's 4 lanes
	OpHAddPS256      // AVX vhaddps ymm: pairwise add of two YMMs
	OpExtractF128Lo  // vextractf128 imm=0: low 128 bits of YMM into XMM
	OpExtractF128Hi  // vextractf128 imm=1: high 128 bits of YMM into XMM

	// Math operations (v0.8) - lowered to libm calls or inline sequences
	OpExpF32   // exp(x) for f32
	OpLogF32   // log(x) for f32
	OpSqrtF32  // sqrt(x) for f32
	OpPowF32   // pow(x, y) for f32
	OpSinF32   // sin(x) for f32
	OpCosF32   // cos(x) for f32
	OpAbsF32   // abs(x) for f32
	OpMinF32   // min(x, y) for f32
	OpMaxF32   // max(x, y) for f32
	OpClampF32 // clamp(x, min, max) for f32

	// Neural network operations (v0.8) - typically lowered to loops or library calls
	OpReLUF32    // max(0, x) for f32
	OpSigmoidF32 // 1 / (1 + exp(-x)) for f32
	OpTanhF32    // tanh(x) for f32

	// Vector operations (v0.8)
	OpDotProduct // dot product of two vectors
	OpNormalize  // normalize vector
	OpCross3     // cross product of 3D vectors
)

type Value struct {
	ID   int
	Type *types.Type
}

type Inst struct {
	Result  Value
	Op      Op
	Imm     int64
	ImmF    float64
	ImmB    bool
	FnName  string
	Args    []Value
	LabelID int     // for OpLabel, OpJumpIf, OpJump
	Aux     []int64 // auxiliary data (e.g., [M, K, N] for matmul)
}
