// Package ceir defines the Core ESque IR — administrative-normal form,
// fully typed. Every operand is a name or literal.
package ceir

import "github.com/esque-lang/esquec/internal/types"

// Module is a CEIR module.
type Module struct {
	Path string
	Fns  []*Func
}

// Func is a CEIR function.
type Func struct {
	Name    string
	Params  []Param
	RetType *types.Type
	Body    []Inst
	// Result is the SSA value produced as the function's result.
	Result Value
}

type Param struct {
	Name string
	Type *types.Type
}

// Op enumerates CEIR operations.
type Op int

const (
	OpInvalid Op = iota
	OpConstInt
	OpConstFloat
	OpConstBool
	OpConstString // String literal: Imm holds length, FnName holds rodata sym (assigned at MIR lower time)
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMod
	OpNeg
	OpNot    // logical not
	OpEq     // ==
	OpNe     // !=
	OpLt     // <
	OpLe     // <=
	OpGt     // >
	OpGe     // >=
	OpAnd    // &&
	OpOr     // ||
	OpCall   // function call
	OpSelect // ternary: select(cond, t, f)
	OpCopy   // copy a value (used for parameter passing)

	// Tensor operations
	OpTensorLit   // create tensor from element values
	OpTensorAdd   // element-wise add
	OpTensorSub   // element-wise sub
	OpTensorMul   // element-wise mul
	OpTensorDiv   // element-wise div
	OpTensorElem  // extract single element: Args[0]=tensor, Imm=index
	OpReduceSum   // sum reduction
	OpReduceProd  // product reduction
	OpReduceMax   // max reduction
	OpReduceMin   // min reduction
	OpMatMul      // matrix multiply: Args[0] @ Args[1]

	// Cast operations
	OpCastF32ToI32 // truncate f32 to i32
	OpCastF64ToI32 // truncate f64 to i32
	OpCastI32ToF32 // convert i32 to f32
	OpCastI32ToF64 // convert i32 to f64

	// Integer width casts involving i8/u8 (v0.12). Sign- or zero-extend
	// the canonical low-byte representation of the source into the
	// canonical 64-bit representation of the destination type.
	OpCastI8ToI32  // sign-extend low byte
	OpCastI8ToI64  // sign-extend low byte
	OpCastU8ToI32  // zero-extend low byte
	OpCastU8ToI64  // zero-extend low byte
	OpCastI32ToI8  // truncate to low byte, sign-extend back
	OpCastI32ToU8  // truncate to low byte, zero-extend back
	OpCastI64ToI8  // truncate to low byte, sign-extend back
	OpCastI64ToU8  // truncate to low byte, zero-extend back

	// Neural network operations (v0.8)
	OpConv2D      // 2D convolution: conv2d(input, kernel)
	OpMaxPool2D   // 2D max pooling: maxpool2d(input, kernel_size)
	OpAvgPool2D   // 2D average pooling: avgpool2d(input, kernel_size)
	OpReLU        // ReLU activation: relu(x) = max(0, x)
	OpSigmoid     // Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))
	OpTanh        // Tanh activation: tanh(x)
	OpSoftmax     // Softmax: softmax(x)
	OpBatchNorm   // Batch normalization
	OpDropout     // Dropout (training only)
	OpFlatten     // Flatten tensor to 1D

	// Math operations (v0.8)
	OpExp         // exp(x)
	OpLog         // log(x)
	OpSqrt        // sqrt(x)
	OpPow         // pow(x, y)
	OpSin         // sin(x)
	OpCos         // cos(x)
	OpClamp       // clamp(x, min, max)
	OpMin         // min(x, y)
	OpMax         // max(x, y)
	OpAbs         // abs(x)

	// Graphics operations (v0.8)
	OpNormalize   // normalize vector
	OpDot         // dot product
	OpCross       // cross product (vec3)
	OpMix         // linear interpolation: mix(a, b, t)
	OpSmoothstep  // smoothstep interpolation

	// Tensor operations (v0.8)
	OpTranspose   // matrix transpose
	OpReshape     // reshape tensor
	OpBroadcast   // broadcast tensor to shape
	OpConcat      // concatenate tensors
	OpSlice       // slice tensor
	OpGather      // gather elements by indices
	OpScatter     // scatter elements by indices

	// Functional iteration operations (v0.9)
	OpFold        // fold/reduce with custom function: Args[0]=tensor, FnName=reducer
	OpFoldInit    // fold with init: Args[0]=init, Args[1]=tensor, FnName=reducer
	OpIterate     // iterate(n, init, f): Args[0]=init, Imm=count, FnName=updater
	OpIterateLoop // loop-based iterate: Args[0]=init, Imm=count, BodyInsts=body, LoopInID/LoopOutID for state

	// Large-N runtime-loop forms (v0.11). Used when the unrolled
	// count would exceed the inline ceiling. Each emits a real
	// counted loop in MIR rather than unrolling at CEIR time.
	OpTabulateLoop // tabulate(N, |i| body): Imm=count, BodyInsts=body, LoopInID=index, LoopOutID=elem
	OpScanLoop     // scan(init, |a,x| body, v): Args[0]=init, Args[1]=tensor, Imm=count, BodyInsts=body, LoopInID=acc, LoopInID2=elem, LoopOutID=newAcc
	OpEachLoop     // each(v, fn): Args[0]=tensor, Imm=count, FnName=callee accepted by the type checker
)

func (o Op) String() string {
	switch o {
	case OpConstInt:
		return "const_int"
	case OpConstFloat:
		return "const_float"
	case OpConstBool:
		return "const_bool"
	case OpConstString:
		return "const_string"
	case OpAdd:
		return "add"
	case OpSub:
		return "sub"
	case OpMul:
		return "mul"
	case OpDiv:
		return "div"
	case OpMod:
		return "mod"
	case OpNeg:
		return "neg"
	case OpNot:
		return "not"
	case OpEq:
		return "eq"
	case OpNe:
		return "ne"
	case OpLt:
		return "lt"
	case OpLe:
		return "le"
	case OpGt:
		return "gt"
	case OpGe:
		return "ge"
	case OpAnd:
		return "and"
	case OpOr:
		return "or"
	case OpCall:
		return "call"
	case OpSelect:
		return "select"
	case OpCopy:
		return "copy"
	case OpTensorLit:
		return "tensor_lit"
	case OpTensorAdd:
		return "tensor_add"
	case OpTensorSub:
		return "tensor_sub"
	case OpTensorMul:
		return "tensor_mul"
	case OpTensorDiv:
		return "tensor_div"
	case OpTensorElem:
		return "tensor_elem"
	case OpReduceSum:
		return "reduce_sum"
	case OpReduceProd:
		return "reduce_prod"
	case OpReduceMax:
		return "reduce_max"
	case OpReduceMin:
		return "reduce_min"
	case OpMatMul:
		return "matmul"
	case OpCastF32ToI32:
		return "cast_f32_to_i32"
	case OpCastF64ToI32:
		return "cast_f64_to_i32"
	case OpCastI32ToF32:
		return "cast_i32_to_f32"
	case OpCastI32ToF64:
		return "cast_i32_to_f64"
	case OpCastI8ToI32:
		return "cast_i8_to_i32"
	case OpCastI8ToI64:
		return "cast_i8_to_i64"
	case OpCastU8ToI32:
		return "cast_u8_to_i32"
	case OpCastU8ToI64:
		return "cast_u8_to_i64"
	case OpCastI32ToI8:
		return "cast_i32_to_i8"
	case OpCastI32ToU8:
		return "cast_i32_to_u8"
	case OpCastI64ToI8:
		return "cast_i64_to_i8"
	case OpCastI64ToU8:
		return "cast_i64_to_u8"
	// Neural network operations
	case OpConv2D:
		return "conv2d"
	case OpMaxPool2D:
		return "maxpool2d"
	case OpAvgPool2D:
		return "avgpool2d"
	case OpReLU:
		return "relu"
	case OpSigmoid:
		return "sigmoid"
	case OpTanh:
		return "tanh"
	case OpSoftmax:
		return "softmax"
	case OpBatchNorm:
		return "batch_norm"
	case OpDropout:
		return "dropout"
	case OpFlatten:
		return "flatten"
	// Math operations
	case OpExp:
		return "exp"
	case OpLog:
		return "log"
	case OpSqrt:
		return "sqrt"
	case OpPow:
		return "pow"
	case OpSin:
		return "sin"
	case OpCos:
		return "cos"
	case OpClamp:
		return "clamp"
	case OpMin:
		return "min"
	case OpMax:
		return "max"
	case OpAbs:
		return "abs"
	// Graphics operations
	case OpNormalize:
		return "normalize"
	case OpDot:
		return "dot"
	case OpCross:
		return "cross"
	case OpMix:
		return "mix"
	case OpSmoothstep:
		return "smoothstep"
	// Tensor operations
	case OpTranspose:
		return "transpose"
	case OpReshape:
		return "reshape"
	case OpBroadcast:
		return "broadcast"
	case OpConcat:
		return "concat"
	case OpSlice:
		return "slice"
	case OpGather:
		return "gather"
	case OpScatter:
		return "scatter"
	// Functional iteration operations
	case OpFold:
		return "fold"
	case OpFoldInit:
		return "fold_init"
	case OpIterate:
		return "iterate"
	case OpIterateLoop:
		return "iterate_loop"
	case OpTabulateLoop:
		return "tabulate_loop"
	case OpScanLoop:
		return "scan_loop"
	case OpEachLoop:
		return "each_loop"
	}
	return "?"
}

// Value is an SSA name produced by an instruction.
type Value struct {
	ID   int
	Type *types.Type
}

// Inst is a single CEIR instruction.
type Inst struct {
	Result Value
	Op     Op
	Imm    int64   // for OpConstInt, OpIterateLoop (count)
	ImmF   float64 // for OpConstFloat
	ImmB   bool    // for OpConstBool
	ImmS   string  // for OpConstString (the literal bytes)
	FnName string  // for OpCall
	Args   []Value // for arithmetic/call ops

	// Loop body fields for OpIterateLoop / OpTabulateLoop / OpScanLoop
	BodyInsts []Inst // nested instructions for loop body
	LoopInID  int    // ID of the loop input value (state for iterate, index for tabulate, acc for scan)
	LoopInID2 int    // second loop input ID (scan only: element from input tensor)
	LoopOutID int    // ID of the loop output value (result of body)
}
