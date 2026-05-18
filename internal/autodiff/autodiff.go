// Package autodiff implements automatic differentiation for neural network training.
// It provides tape-based reverse-mode AD (backpropagation).
package autodiff

import (
	"fmt"

	"github.com/esque-lang/esquec/internal/ceir"
	"github.com/esque-lang/esquec/internal/types"
)

// Tape records operations for reverse-mode automatic differentiation.
type Tape struct {
	nodes   []*Node
	gradMap map[int]*ceir.Value // value ID -> gradient value
	nextID  int
}

// Node represents an operation in the computation graph.
type Node struct {
	ID       int
	Op       ceir.Op
	Inputs   []int      // IDs of input nodes
	Output   ceir.Value // The forward pass output
	GradFunc GradFunc   // Function to compute gradients
}

// GradFunc computes gradients given the output gradient.
// Returns gradients for each input.
type GradFunc func(outputGrad ceir.Value, inputs []ceir.Value, output ceir.Value, ctx *GradContext) []ceir.Value

// GradContext provides utilities for gradient computation.
type GradContext struct {
	tape   *Tape
	insts  *[]ceir.Inst
	nextID *int
}

// NewTape creates a new computation tape.
func NewTape() *Tape {
	return &Tape{
		gradMap: make(map[int]*ceir.Value),
	}
}

// Record adds an operation to the tape.
func (t *Tape) Record(op ceir.Op, inputs []ceir.Value, output ceir.Value) *Node {
	node := &Node{
		ID:       t.nextID,
		Op:       op,
		Inputs:  make([]int, len(inputs)),
		Output:  output,
		GradFunc: getGradFunc(op),
	}
	for i, in := range inputs {
		node.Inputs[i] = in.ID
	}
	t.nextID++
	t.nodes = append(t.nodes, node)
	return node
}

// Backward computes gradients for all recorded operations.
// loss is the value to differentiate from (typically scalar loss).
// Returns a map from value ID to gradient instructions.
func (t *Tape) Backward(loss ceir.Value) ([]ceir.Inst, error) {
	var insts []ceir.Inst
	nextID := loss.ID + 1000 // Start gradient IDs high to avoid collisions

	ctx := &GradContext{
		tape:   t,
		insts:  &insts,
		nextID: &nextID,
	}

	// Initialize gradient of loss to 1.0
	lossGrad := ctx.fresh(loss.Type)
	insts = append(insts, ceir.Inst{
		Result: lossGrad,
		Op:     ceir.OpConstFloat,
		ImmF:   1.0,
	})
	t.gradMap[loss.ID] = &lossGrad

	// Process nodes in reverse order (backpropagation)
	for i := len(t.nodes) - 1; i >= 0; i-- {
		node := t.nodes[i]

		// Get output gradient
		outputGrad, ok := t.gradMap[node.Output.ID]
		if !ok {
			// No gradient flows to this node
			continue
		}

		// Skip if no gradient function
		if node.GradFunc == nil {
			continue
		}

		// Collect input values
		inputs := make([]ceir.Value, len(node.Inputs))
		for j, inputID := range node.Inputs {
			for _, n := range t.nodes {
				if n.Output.ID == inputID {
					inputs[j] = n.Output
					break
				}
			}
		}

		// Compute input gradients
		inputGrads := node.GradFunc(*outputGrad, inputs, node.Output, ctx)

		// Accumulate gradients
		for j, inputID := range node.Inputs {
			if j >= len(inputGrads) {
				break
			}
			grad := inputGrads[j]
			if existing, ok := t.gradMap[inputID]; ok {
				// Accumulate: grad = existing + new
				acc := ctx.fresh(grad.Type)
				insts = append(insts, ceir.Inst{
					Result: acc,
					Op:     ceir.OpAdd,
					Args:   []ceir.Value{*existing, grad},
				})
				t.gradMap[inputID] = &acc
			} else {
				t.gradMap[inputID] = &grad
			}
		}
	}

	return insts, nil
}

// GetGradient returns the gradient for a value.
func (t *Tape) GetGradient(id int) (*ceir.Value, bool) {
	g, ok := t.gradMap[id]
	return g, ok
}

// fresh creates a fresh value ID.
func (ctx *GradContext) fresh(t *types.Type) ceir.Value {
	id := *ctx.nextID
	*ctx.nextID++
	return ceir.Value{ID: id, Type: t}
}

// emit adds an instruction and returns the result.
func (ctx *GradContext) emit(op ceir.Op, t *types.Type, args ...ceir.Value) ceir.Value {
	result := ctx.fresh(t)
	*ctx.insts = append(*ctx.insts, ceir.Inst{
		Result: result,
		Op:     op,
		Args:   args,
	})
	return result
}

// emitConst emits a float constant.
func (ctx *GradContext) emitConst(t *types.Type, val float64) ceir.Value {
	result := ctx.fresh(t)
	*ctx.insts = append(*ctx.insts, ceir.Inst{
		Result: result,
		Op:     ceir.OpConstFloat,
		ImmF:   val,
	})
	return result
}

// getGradFunc returns the gradient function for an operation.
func getGradFunc(op ceir.Op) GradFunc {
	switch op {
	case ceir.OpAdd:
		return gradAdd
	case ceir.OpSub:
		return gradSub
	case ceir.OpMul:
		return gradMul
	case ceir.OpDiv:
		return gradDiv
	case ceir.OpNeg:
		return gradNeg
	case ceir.OpExp:
		return gradExp
	case ceir.OpLog:
		return gradLog
	case ceir.OpSqrt:
		return gradSqrt
	case ceir.OpSin:
		return gradSin
	case ceir.OpCos:
		return gradCos
	case ceir.OpPow:
		return gradPow
	case ceir.OpReLU:
		return gradReLU
	case ceir.OpSigmoid:
		return gradSigmoid
	case ceir.OpTanh:
		return gradTanh
	case ceir.OpMatMul:
		return gradMatMul
	case ceir.OpTensorAdd, ceir.OpTensorSub, ceir.OpTensorMul, ceir.OpTensorDiv:
		return gradTensorElementwise(op)
	case ceir.OpReduceSum:
		return gradReduceSum
	default:
		return nil
	}
}

// Gradient implementations

// gradAdd: d/dx (x + y) = 1, d/dy (x + y) = 1
func gradAdd(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	// Both inputs get the same gradient
	return []ceir.Value{outGrad, outGrad}
}

// gradSub: d/dx (x - y) = 1, d/dy (x - y) = -1
func gradSub(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	negGrad := ctx.emit(ceir.OpNeg, outGrad.Type, outGrad)
	return []ceir.Value{outGrad, negGrad}
}

// gradMul: d/dx (x * y) = y, d/dy (x * y) = x
func gradMul(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 2 {
		return nil
	}
	gradX := ctx.emit(ceir.OpMul, outGrad.Type, outGrad, inputs[1])
	gradY := ctx.emit(ceir.OpMul, outGrad.Type, outGrad, inputs[0])
	return []ceir.Value{gradX, gradY}
}

// gradDiv: d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
func gradDiv(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 2 {
		return nil
	}
	x, y := inputs[0], inputs[1]
	// d/dx = outGrad / y
	gradX := ctx.emit(ceir.OpDiv, outGrad.Type, outGrad, y)
	// d/dy = -outGrad * x / y^2
	y2 := ctx.emit(ceir.OpMul, y.Type, y, y)
	xOverY2 := ctx.emit(ceir.OpDiv, x.Type, x, y2)
	negXOverY2 := ctx.emit(ceir.OpNeg, xOverY2.Type, xOverY2)
	gradY := ctx.emit(ceir.OpMul, outGrad.Type, outGrad, negXOverY2)
	return []ceir.Value{gradX, gradY}
}

// gradNeg: d/dx (-x) = -1
func gradNeg(outGrad ceir.Value, _ []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	return []ceir.Value{ctx.emit(ceir.OpNeg, outGrad.Type, outGrad)}
}

// gradExp: d/dx exp(x) = exp(x)
func gradExp(outGrad ceir.Value, _ []ceir.Value, output ceir.Value, ctx *GradContext) []ceir.Value {
	// exp'(x) = exp(x), and output = exp(x)
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, output)}
}

// gradLog: d/dx log(x) = 1/x
func gradLog(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 1 {
		return nil
	}
	one := ctx.emitConst(inputs[0].Type, 1.0)
	invX := ctx.emit(ceir.OpDiv, inputs[0].Type, one, inputs[0])
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, invX)}
}

// gradSqrt: d/dx sqrt(x) = 1 / (2 * sqrt(x))
func gradSqrt(outGrad ceir.Value, _ []ceir.Value, output ceir.Value, ctx *GradContext) []ceir.Value {
	// output = sqrt(x)
	two := ctx.emitConst(output.Type, 2.0)
	twoSqrtX := ctx.emit(ceir.OpMul, output.Type, two, output)
	one := ctx.emitConst(output.Type, 1.0)
	invTwoSqrtX := ctx.emit(ceir.OpDiv, output.Type, one, twoSqrtX)
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, invTwoSqrtX)}
}

// gradSin: d/dx sin(x) = cos(x)
func gradSin(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 1 {
		return nil
	}
	cosX := ctx.emit(ceir.OpCos, inputs[0].Type, inputs[0])
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, cosX)}
}

// gradCos: d/dx cos(x) = -sin(x)
func gradCos(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 1 {
		return nil
	}
	sinX := ctx.emit(ceir.OpSin, inputs[0].Type, inputs[0])
	negSinX := ctx.emit(ceir.OpNeg, sinX.Type, sinX)
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, negSinX)}
}

// gradPow: d/dx x^y = y * x^(y-1), d/dy x^y = x^y * log(x)
func gradPow(outGrad ceir.Value, inputs []ceir.Value, output ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 2 {
		return nil
	}
	x, y := inputs[0], inputs[1]

	// d/dx = outGrad * y * x^(y-1) = outGrad * y * output / x
	yTimesOut := ctx.emit(ceir.OpMul, y.Type, y, output)
	gradX := ctx.emit(ceir.OpDiv, x.Type, yTimesOut, x)
	gradX = ctx.emit(ceir.OpMul, outGrad.Type, outGrad, gradX)

	// d/dy = outGrad * x^y * log(x) = outGrad * output * log(x)
	logX := ctx.emit(ceir.OpLog, x.Type, x)
	outTimesLogX := ctx.emit(ceir.OpMul, output.Type, output, logX)
	gradY := ctx.emit(ceir.OpMul, outGrad.Type, outGrad, outTimesLogX)

	return []ceir.Value{gradX, gradY}
}

// gradReLU: d/dx relu(x) = 1 if x > 0, else 0
func gradReLU(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 1 {
		return nil
	}
	x := inputs[0]
	zero := ctx.emitConst(x.Type, 0.0)
	one := ctx.emitConst(x.Type, 1.0)
	// mask = x > 0 ? 1 : 0
	cond := ctx.emit(ceir.OpGt, types.Bool(), x, zero)
	mask := ctx.emit(ceir.OpSelect, x.Type, cond, one, zero)
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, mask)}
}

// gradSigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
func gradSigmoid(outGrad ceir.Value, _ []ceir.Value, output ceir.Value, ctx *GradContext) []ceir.Value {
	// output = sigmoid(x)
	one := ctx.emitConst(output.Type, 1.0)
	oneMinusOut := ctx.emit(ceir.OpSub, output.Type, one, output)
	sigmoidDeriv := ctx.emit(ceir.OpMul, output.Type, output, oneMinusOut)
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, sigmoidDeriv)}
}

// gradTanh: d/dx tanh(x) = 1 - tanh(x)^2
func gradTanh(outGrad ceir.Value, _ []ceir.Value, output ceir.Value, ctx *GradContext) []ceir.Value {
	// output = tanh(x)
	outSquared := ctx.emit(ceir.OpMul, output.Type, output, output)
	one := ctx.emitConst(output.Type, 1.0)
	oneMinusOutSq := ctx.emit(ceir.OpSub, output.Type, one, outSquared)
	return []ceir.Value{ctx.emit(ceir.OpMul, outGrad.Type, outGrad, oneMinusOutSq)}
}

// gradMatMul: d/dA (A @ B) = dOut @ B^T, d/dB (A @ B) = A^T @ dOut
func gradMatMul(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 2 {
		return nil
	}
	A, B := inputs[0], inputs[1]

	// gradA = outGrad @ B^T
	BT := ctx.emit(ceir.OpTranspose, B.Type, B)
	gradA := ctx.emit(ceir.OpMatMul, A.Type, outGrad, BT)

	// gradB = A^T @ outGrad
	AT := ctx.emit(ceir.OpTranspose, A.Type, A)
	gradB := ctx.emit(ceir.OpMatMul, B.Type, AT, outGrad)

	return []ceir.Value{gradA, gradB}
}

// gradTensorElementwise handles element-wise tensor operations.
func gradTensorElementwise(op ceir.Op) GradFunc {
	switch op {
	case ceir.OpTensorAdd:
		return func(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, _ *GradContext) []ceir.Value {
			return []ceir.Value{outGrad, outGrad}
		}
	case ceir.OpTensorSub:
		return func(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
			negGrad := ctx.emit(ceir.OpNeg, outGrad.Type, outGrad)
			return []ceir.Value{outGrad, negGrad}
		}
	case ceir.OpTensorMul:
		return func(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
			if len(inputs) < 2 {
				return nil
			}
			gradA := ctx.emit(ceir.OpTensorMul, outGrad.Type, outGrad, inputs[1])
			gradB := ctx.emit(ceir.OpTensorMul, outGrad.Type, outGrad, inputs[0])
			return []ceir.Value{gradA, gradB}
		}
	case ceir.OpTensorDiv:
		return func(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
			if len(inputs) < 2 {
				return nil
			}
			// gradA = outGrad / B
			gradA := ctx.emit(ceir.OpTensorDiv, outGrad.Type, outGrad, inputs[1])
			// gradB = -outGrad * A / B^2
			B2 := ctx.emit(ceir.OpTensorMul, inputs[1].Type, inputs[1], inputs[1])
			AOverB2 := ctx.emit(ceir.OpTensorDiv, inputs[0].Type, inputs[0], B2)
			negAOverB2 := ctx.emit(ceir.OpNeg, AOverB2.Type, AOverB2)
			gradB := ctx.emit(ceir.OpTensorMul, outGrad.Type, outGrad, negAOverB2)
			return []ceir.Value{gradA, gradB}
		}
	}
	return nil
}

// gradReduceSum: d/dx sum(x) = ones_like(x) * outGrad
func gradReduceSum(outGrad ceir.Value, inputs []ceir.Value, _ ceir.Value, ctx *GradContext) []ceir.Value {
	if len(inputs) < 1 {
		return nil
	}
	// Broadcast the scalar gradient to the input shape
	grad := ctx.emit(ceir.OpBroadcast, inputs[0].Type, outGrad)
	return []ceir.Value{grad}
}

// Differentiate computes the gradient of a CEIR function with respect to its inputs.
// Returns a new function that computes both the forward pass and gradients.
func Differentiate(fn *ceir.Func) (*ceir.Func, error) {
	tape := NewTape()

	// Record all operations
	for _, inst := range fn.Body {
		if len(inst.Args) > 0 {
			tape.Record(inst.Op, inst.Args, inst.Result)
		}
	}

	// Compute gradients from the result
	gradInsts, err := tape.Backward(fn.Result)
	if err != nil {
		return nil, fmt.Errorf("backward pass: %w", err)
	}

	// Create new function with both forward and backward passes
	newFn := &ceir.Func{
		Name:    fn.Name + "_grad",
		Params:  fn.Params,
		RetType: fn.RetType,
		Body:    append(fn.Body, gradInsts...),
		Result:  fn.Result,
	}

	return newFn, nil
}
