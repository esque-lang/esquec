package mir

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/esque-lang/esquec/internal/ceir"
	"github.com/esque-lang/esquec/internal/types"
)

// lowerCtx holds state for lowering a single function.
type lowerCtx struct {
	nextID    int
	nextLabel int
	insts     []Inst
	mod       *Module // module-level rodata sink; nil-safe for tests

	// constTensors maps a CEIR value ID to its flattened constant bytes
	// when the value is the result of a fully-constant OpTensorLit. The
	// OpTensorLit lowering checks this map before emitting stack stores
	// so that pure-literal tensors land in .rodata instead.
	constTensors map[int][]byte
	// constScalars maps a CEIR value ID to its scalar constant value as
	// raw little-endian bytes (4 bytes per element for f32/i32). Used
	// only as input to constTensors construction.
	constScalars map[int][]byte

	// stringLengths records, for every CEIR value of type string that
	// has been lowered to a (rodata ptr, len) pair, the constant length
	// in bytes. The MIR Value stored in valMap holds the pointer; the
	// length is recovered from this map at print_str call sites so a
	// single CEIR string arg can be expanded into the (ptr, len)
	// two-arg ABI without adding multi-Value SSA nodes.
	stringLengths map[int]int64
}

func (c *lowerCtx) fresh(t *types.Type) Value {
	v := Value{ID: c.nextID, Type: t}
	c.nextID++
	return v
}

func (c *lowerCtx) freshLabel() int {
	l := c.nextLabel
	c.nextLabel++
	return l
}

func (c *lowerCtx) emit(in Inst) {
	c.insts = append(c.insts, in)
}

// LowerFromCEIR converts CEIR to MIR, expanding tensor ops to loops.
func LowerFromCEIR(m *ceir.Module) *Module {
	out := &Module{Path: m.Path}
	for _, f := range m.Fns {
		out.Fns = append(out.Fns, lowerFn(f, out))
	}
	return out
}

func lowerFn(f *ceir.Func, mod *Module) *Func {
	// Start IDs after parameters
	ctx := &lowerCtx{
		nextID:        len(f.Params),
		mod:           mod,
		constTensors:  make(map[int][]byte),
		constScalars:  make(map[int][]byte),
		stringLengths: make(map[int]int64),
	}

	// Pre-scan: identify CEIR values that are pure constants and tensor
	// literals built entirely from constants. This drives the .rodata
	// fast path in OpTensorLit lowering below.
	ctx.scanConstants(f)

	// A CEIR scalar constant whose ONLY consumers are tensor literals
	// that became .rodata blobs has no live consumer in MIR — the
	// store sequence that would have read it was elided. Lowering it
	// anyway emits a dead OpConstFloat that pins an XMM register
	// (which aliases a YMM lane), starving subsequent SIMD ops. Find
	// such absorbed constants and skip their lowering.
	absorbed := computeAbsorbedConsts(f, ctx.constTensors)

	// Map from CEIR value IDs to MIR value IDs (may differ for tensor ops)
	valMap := make(map[int]Value)

	// Map parameters
	for i, p := range f.Params {
		valMap[i] = Value{ID: i, Type: p.Type}
	}

	// Process each instruction
	for _, in := range f.Body {
		if absorbed[in.Result.ID] {
			continue
		}
		ctx.lowerInst(in, valMap)
	}

	var ps []Param
	for _, p := range f.Params {
		ps = append(ps, Param{Name: p.Name, Type: p.Type})
	}

	return &Func{
		Name:    f.Name,
		Params:  ps,
		RetType: f.RetType,
		Body:    ctx.insts,
		Result:  valMap[f.Result.ID],
	}
}

func (c *lowerCtx) lowerInst(in ceir.Inst, valMap map[int]Value) {
	switch in.Op {
	case ceir.OpTensorLit:
		// Fast path: if every leaf of this literal is a compile-time
		// constant, intern the flattened bytes in .rodata and emit a
		// single OpRodataPtr that loads the address. This skips the
		// stack alloc + per-element store sequence entirely.
		if blob, ok := c.constTensors[in.Result.ID]; ok && c.mod != nil {
			sym := fmt.Sprintf("__rodata_T%d", len(c.mod.Rodata))
			c.mod.Rodata = append(c.mod.Rodata, RodataEntry{
				Sym: sym, Bytes: blob, Align: 16,
			})
			ptr := c.fresh(types.I64())
			c.emit(Inst{Result: ptr, Op: OpRodataPtr, FnName: sym})
			valMap[in.Result.ID] = ptr
			return
		}

		// Allocate stack space for tensor
		// For rank-2+ tensors, args are sub-tensor base addresses
		tensorType := in.Result.Type

		// Calculate total elements
		totalElements := 1
		for _, dim := range tensorType.Shape {
			totalElements *= int(dim.Val)
		}
		bytes := int64(totalElements * 4) // f32 = 4 bytes

		// Create stack allocation
		baseVal := c.fresh(types.I64()) // pointer type
		c.emit(Inst{Result: baseVal, Op: OpStackAlloc, Imm: bytes})

		// Check if this is a nested tensor literal (args are tensor pointers)
		if len(in.Args) > 0 && in.Args[0].Type.IsTensor() {
			// Rank-2+ tensor: each arg is a pointer to a sub-tensor
			// Copy each sub-tensor's data into the result
			innerElements := 1
			for i := 1; i < len(tensorType.Shape); i++ {
				innerElements *= int(tensorType.Shape[i].Val)
			}

			for i, arg := range in.Args {
				srcBase := valMap[arg.ID]
				dstOffset := int64(i * innerElements * 4)

				// Copy innerElements floats from srcBase to baseVal+dstOffset
				for j := 0; j < innerElements; j++ {
					elemOffset := int64(j * 4)
					elem := c.fresh(tensorType.ElemType())
					c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{srcBase}, Imm: elemOffset})
					c.emit(Inst{Op: OpStore32, Args: []Value{baseVal, elem}, Imm: dstOffset + elemOffset})
				}
			}
		} else {
			// Rank-1 tensor: args are scalar values. Dispatch on element
			// type so i32 stores go through a GPR path.
			storeOp := OpStore32
			if elemType := tensorType.ElemType(); elemType != nil && elemType.K == types.KI32 {
				storeOp = OpStoreI32
			}
			for i, arg := range in.Args {
				srcVal := valMap[arg.ID]
				offset := int64(i * 4)
				c.emit(Inst{Op: storeOp, Args: []Value{baseVal, srcVal}, Imm: offset})
			}
		}

		// The tensor value is represented by its base address
		valMap[in.Result.ID] = baseVal

	case ceir.OpTensorAdd, ceir.OpTensorSub, ceir.OpTensorMul, ceir.OpTensorDiv:
		// Element-wise operation: a .op b
		aBase := valMap[in.Args[0].ID]
		bBase := valMap[in.Args[1].ID]

		// Get total tensor size from type (works for any rank)
		tensorType := in.Result.Type
		totalElements := 1
		for _, dim := range tensorType.Shape {
			totalElements *= int(dim.Val)
		}

		// Allocate result
		bytes := int64(totalElements * 4)
		resultBase := c.fresh(types.I64())
		c.emit(Inst{Result: resultBase, Op: OpStackAlloc, Imm: bytes})

		// Determine the scalar, SSE, and AVX2 operations
		var floatOp, packedOp128, packedOp256 Op
		switch in.Op {
		case ceir.OpTensorAdd:
			floatOp = OpAddF32
			packedOp128 = OpAddPS128
			packedOp256 = OpAddPS256
		case ceir.OpTensorSub:
			floatOp = OpSubF32
			packedOp128 = OpSubPS128
			packedOp256 = OpSubPS256
		case ceir.OpTensorMul:
			floatOp = OpMulF32
			packedOp128 = OpMulPS128
			packedOp256 = OpMulPS256
		case ceir.OpTensorDiv:
			floatOp = OpDivF32
			packedOp128 = OpDivPS128
			packedOp256 = OpDivPS256
		}

		elemType := tensorType.ElemType()

		// Hybrid vectorization: AVX2 (8) + SSE (4) + scalar cleanup
		remaining := totalElements
		offset := int64(0)

		// AVX2: process 8 elements at a time
		for remaining >= 8 {
			aVec := c.fresh(types.Vec8F32())
			c.emit(Inst{Result: aVec, Op: OpLoad256, Args: []Value{aBase}, Imm: offset})

			bVec := c.fresh(types.Vec8F32())
			c.emit(Inst{Result: bVec, Op: OpLoad256, Args: []Value{bBase}, Imm: offset})

			rVec := c.fresh(types.Vec8F32())
			c.emit(Inst{Result: rVec, Op: packedOp256, Args: []Value{aVec, bVec}})

			c.emit(Inst{Op: OpStore256, Args: []Value{resultBase, rVec}, Imm: offset})

			offset += 32 // 8 elements * 4 bytes
			remaining -= 8
		}

		// SSE: process 4 elements at a time
		for remaining >= 4 {
			aVec := c.fresh(types.Vec4F32())
			c.emit(Inst{Result: aVec, Op: OpLoad128, Args: []Value{aBase}, Imm: offset})

			bVec := c.fresh(types.Vec4F32())
			c.emit(Inst{Result: bVec, Op: OpLoad128, Args: []Value{bBase}, Imm: offset})

			rVec := c.fresh(types.Vec4F32())
			c.emit(Inst{Result: rVec, Op: packedOp128, Args: []Value{aVec, bVec}})

			c.emit(Inst{Op: OpStore128, Args: []Value{resultBase, rVec}, Imm: offset})

			offset += 16 // 4 elements * 4 bytes
			remaining -= 4
		}

		// Scalar: cleanup remaining 0-3 elements
		for remaining > 0 {
			aElem := c.fresh(elemType)
			c.emit(Inst{Result: aElem, Op: OpLoad32, Args: []Value{aBase}, Imm: offset})

			bElem := c.fresh(elemType)
			c.emit(Inst{Result: bElem, Op: OpLoad32, Args: []Value{bBase}, Imm: offset})

			rElem := c.fresh(elemType)
			c.emit(Inst{Result: rElem, Op: floatOp, Args: []Value{aElem, bElem}})

			c.emit(Inst{Op: OpStore32, Args: []Value{resultBase, rElem}, Imm: offset})

			offset += 4
			remaining--
		}

		valMap[in.Result.ID] = resultBase

	case ceir.OpTensorElem:
		// Extract single element from tensor: Args[0]=tensor, Imm=index
		tensorBase := valMap[in.Args[0].ID]
		offset := in.Imm * 4 // index * 4 bytes (i32 or f32)
		elem := c.fresh(in.Result.Type)
		// Dispatch load on element type so i32 tensors use a GPR load.
		loadOp := OpLoad32
		if in.Result.Type != nil && in.Result.Type.K == types.KI32 {
			loadOp = OpLoadI32
		}
		c.emit(Inst{Result: elem, Op: loadOp, Args: []Value{tensorBase}, Imm: offset})
		valMap[in.Result.ID] = elem

	case ceir.OpReduceSum, ceir.OpReduceProd, ceir.OpReduceMax, ceir.OpReduceMin:
		// Reduction: +/ tensor, */ tensor, max/ tensor, min/ tensor
		// For rank-1: reduces to scalar
		// For rank-2+: reduces last dimension, result is rank-1 lower
		tensorBase := valMap[in.Args[0].ID]
		tensorType := in.Args[0].Type
		elemType := tensorType.ElemType()

		// Determine operation and initial value
		var floatOp Op
		var initVal float64
		var useFirstElem bool // for max/min, we use first element as init
		switch in.Op {
		case ceir.OpReduceSum:
			floatOp = OpAddF32
			initVal = 0.0
		case ceir.OpReduceProd:
			floatOp = OpMulF32
			initVal = 1.0
		case ceir.OpReduceMax:
			floatOp = OpMaxF32
			useFirstElem = true
		case ceir.OpReduceMin:
			floatOp = OpMinF32
			useFirstElem = true
		}

		if tensorType.Rank() == 1 {
			// Rank-1: reduce to scalar
			n := int(tensorType.Shape[0].Val)

			// SIMD fast path for sum-reduction at exactly 8 or 4 elements.
			// Uses haddps (and vextractf128 for the AVX2 case) to collapse a
			// packed register to a scalar in lane 0 of an XMM, avoiding a
			// chain of dependent scalar adds. Only applies to OpReduceSum
			// because haddps is an addition; max/min/prod still use the
			// scalar chain below.
			// Strip-mined SIMD reduce-sum for any N >= 8.
			// Build an unrolled YMM accumulator chain over the n/8 chunks,
			// collapse the YMM to a scalar with a horizontal reduction,
			// then add a scalar tail for the remaining n%8 elements.
			// Each YMM chunk is 32 bytes; chunk i is loaded from offset
			// i*32. Result lives in lane 0 of an XMM and feeds scalar f32
			// ops directly.
			if in.Op == ceir.OpReduceSum && n >= 8 {
				chunks := n / 8
				tail := n % 8
				accY := c.fresh(types.Vec8F32())
				c.emit(Inst{Result: accY, Op: OpLoad256, Args: []Value{tensorBase}, Imm: 0})
				for i := 1; i < chunks; i++ {
					next := c.fresh(types.Vec8F32())
					c.emit(Inst{Result: next, Op: OpLoad256, Args: []Value{tensorBase}, Imm: int64(i * 32)})
					sum := c.fresh(types.Vec8F32())
					c.emit(Inst{Result: sum, Op: OpAddPS256, Args: []Value{accY, next}})
					accY = sum
				}
				lo := c.fresh(types.Vec4F32())
				c.emit(Inst{Result: lo, Op: OpExtractF128Lo, Args: []Value{accY}})
				hi := c.fresh(types.Vec4F32())
				c.emit(Inst{Result: hi, Op: OpExtractF128Hi, Args: []Value{accY}})
				sum4 := c.fresh(types.Vec4F32())
				c.emit(Inst{Result: sum4, Op: OpAddPS128, Args: []Value{lo, hi}})
				h1 := c.fresh(types.Vec4F32())
				c.emit(Inst{Result: h1, Op: OpHAddPS128, Args: []Value{sum4, sum4}})
				acc := c.fresh(elemType)
				c.emit(Inst{Result: acc, Op: OpHAddPS128, Args: []Value{h1, h1}})

				// Scalar tail: add the trailing tail elements one at a time.
				tailBase := chunks * 32
				for i := 0; i < tail; i++ {
					elem := c.fresh(elemType)
					c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{tensorBase}, Imm: int64(tailBase + i*4)})
					next := c.fresh(elemType)
					c.emit(Inst{Result: next, Op: OpAddF32, Args: []Value{acc, elem}})
					acc = next
				}

				valMap[in.Result.ID] = acc
				break
			}
			// SSE3 fast path for 4 <= N < 8: collapse the 4-element prefix
			// with two HAddPS128, then add the trailing N%4 elements as a
			// scalar tail. Same shape as the N>=8 strip-mined path but
			// using XMM (128-bit) instead of YMM (256-bit).
			if in.Op == ceir.OpReduceSum && n >= 4 && n < 8 {
				xmm := c.fresh(types.Vec4F32())
				c.emit(Inst{Result: xmm, Op: OpLoad128, Args: []Value{tensorBase}, Imm: 0})
				h1 := c.fresh(types.Vec4F32())
				c.emit(Inst{Result: h1, Op: OpHAddPS128, Args: []Value{xmm, xmm}})
				acc := c.fresh(elemType)
				c.emit(Inst{Result: acc, Op: OpHAddPS128, Args: []Value{h1, h1}})

				tail := n - 4
				tailBase := 16
				for i := 0; i < tail; i++ {
					elem := c.fresh(elemType)
					c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{tensorBase}, Imm: int64(tailBase + i*4)})
					next := c.fresh(elemType)
					c.emit(Inst{Result: next, Op: OpAddF32, Args: []Value{acc, elem}})
					acc = next
				}

				valMap[in.Result.ID] = acc
				break
			}

			var acc Value
			startIdx := 0

			if useFirstElem {
				// For max/min, use first element as initial accumulator
				acc = c.fresh(elemType)
				c.emit(Inst{Result: acc, Op: OpLoad32, Args: []Value{tensorBase}, Imm: 0})
				startIdx = 1
			} else {
				acc = c.fresh(elemType)
				c.emit(Inst{Result: acc, Op: OpConstFloat, ImmF: initVal})
			}

			// Unroll for tensors up to 16 elements (covers AVX2 cases)
			if n <= 16 {
				for i := startIdx; i < n; i++ {
					offset := int64(i * 4)
					elem := c.fresh(elemType)
					c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{tensorBase}, Imm: offset})
					newAcc := c.fresh(elemType)
					c.emit(Inst{Result: newAcc, Op: floatOp, Args: []Value{acc, elem}})
					acc = newAcc
				}
			} else {
				acc = c.emitReductionLoop(tensorBase, acc, n-startIdx, floatOp, elemType)
			}
			valMap[in.Result.ID] = acc
		} else {
			// Rank-2+: reduce last dimension
			// e.g., f32[M,N] -> f32[M] where each output[i] = sum of input[i,0..N-1]
			outerDims := 1
			for i := 0; i < tensorType.Rank()-1; i++ {
				outerDims *= int(tensorType.Shape[i].Val)
			}
			innerDim := int(tensorType.Shape[tensorType.Rank()-1].Val)

			// Allocate result array
			resultBytes := int64(outerDims * 4)
			resultBase := c.fresh(types.I64())
			c.emit(Inst{Result: resultBase, Op: OpStackAlloc, Imm: resultBytes})

			// For each outer index, reduce the inner dimension
			// Unroll if small enough
			if outerDims*innerDim <= 16 {
				for i := 0; i < outerDims; i++ {
					// Initialize accumulator
					acc := c.fresh(elemType)
					c.emit(Inst{Result: acc, Op: OpConstFloat, ImmF: initVal})

					// Sum over inner dimension
					for j := 0; j < innerDim; j++ {
						srcOffset := int64((i*innerDim + j) * 4)
						elem := c.fresh(elemType)
						c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{tensorBase}, Imm: srcOffset})
						newAcc := c.fresh(elemType)
						c.emit(Inst{Result: newAcc, Op: floatOp, Args: []Value{acc, elem}})
						acc = newAcc
					}

					// Store result
					dstOffset := int64(i * 4)
					c.emit(Inst{Op: OpStore32, Args: []Value{resultBase, acc}, Imm: dstOffset})
				}
			} else {
				// TODO: emit nested loops for larger tensors
				// For now, still unroll (may be slow for large tensors)
				for i := 0; i < outerDims; i++ {
					acc := c.fresh(elemType)
					c.emit(Inst{Result: acc, Op: OpConstFloat, ImmF: initVal})
					for j := 0; j < innerDim; j++ {
						srcOffset := int64((i*innerDim + j) * 4)
						elem := c.fresh(elemType)
						c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{tensorBase}, Imm: srcOffset})
						newAcc := c.fresh(elemType)
						c.emit(Inst{Result: newAcc, Op: floatOp, Args: []Value{acc, elem}})
						acc = newAcc
					}
					dstOffset := int64(i * 4)
					c.emit(Inst{Op: OpStore32, Args: []Value{resultBase, acc}, Imm: dstOffset})
				}
			}
			valMap[in.Result.ID] = resultBase
		}

	case ceir.OpMatMul:
		// Matrix multiply: A[M,K] @ B[K,N] -> C[M,N]
		aBase := valMap[in.Args[0].ID]
		bBase := valMap[in.Args[1].ID]

		aType := in.Args[0].Type
		bType := in.Args[1].Type
		M := int(aType.Shape[0].Val)
		K := int(aType.Shape[1].Val)
		N := int(bType.Shape[1].Val)

		elemType := aType.ElemType()

		// Allocate result C[M,N]
		resultBytes := int64(M * N * 4)
		cBase := c.fresh(types.I64())
		c.emit(Inst{Result: cBase, Op: OpStackAlloc, Imm: resultBytes})

		// Triple nested loop: C[i,j] = sum_k A[i,k] * B[k,j]
		// Unroll for small matrices (M,N,K all <= 4)
		if M <= 4 && N <= 4 && K <= 4 {
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					// Initialize accumulator
					acc := c.fresh(elemType)
					c.emit(Inst{Result: acc, Op: OpConstFloat, ImmF: 0.0})

					// Inner loop over k
					for k := 0; k < K; k++ {
						// A[i,k] at offset (i*K + k) * 4
						aOffset := int64((i*K + k) * 4)
						aElem := c.fresh(elemType)
						c.emit(Inst{Result: aElem, Op: OpLoad32, Args: []Value{aBase}, Imm: aOffset})

						// B[k,j] at offset (k*N + j) * 4
						bOffset := int64((k*N + j) * 4)
						bElem := c.fresh(elemType)
						c.emit(Inst{Result: bElem, Op: OpLoad32, Args: []Value{bBase}, Imm: bOffset})

						// acc += a * b
						prod := c.fresh(elemType)
						c.emit(Inst{Result: prod, Op: OpMulF32, Args: []Value{aElem, bElem}})
						newAcc := c.fresh(elemType)
						c.emit(Inst{Result: newAcc, Op: OpAddF32, Args: []Value{acc, prod}})
						acc = newAcc
					}

					// Store C[i,j] at offset (i*N + j) * 4
					cOffset := int64((i*N + j) * 4)
					c.emit(Inst{Op: OpStore32, Args: []Value{cBase, acc}, Imm: cOffset})
				}
			}
		} else {
			// Generate actual loops for larger matrices
			c.emitMatMulLoops(aBase, bBase, cBase, M, K, N, elemType)
		}

		valMap[in.Result.ID] = cBase

	case ceir.OpIterateLoop:
		// Loop-based iteration: iterate(n, init, f) where f is a lambda
		// Args[0] = initial state
		// Imm = iteration count
		// BodyInsts = loop body instructions
		// LoopInID = ID of loop input value
		// LoopOutID = ID of loop output value
		initState := valMap[in.Args[0].ID]
		count := int(in.Imm)

		// We need to use memory for the mutable state since we're in SSA form
		// Allocate stack space for the state
		stateType := in.Result.Type
		var stateBytes int64
		if stateType.IsTensor() {
			totalElements := 1
			for _, dim := range stateType.Shape {
				totalElements *= int(dim.Val)
			}
			stateBytes = int64(totalElements * 4)
		} else {
			stateBytes = 8 // scalar (f32 or i64)
		}

		// Allocate state storage
		statePtr := c.fresh(types.I64())
		c.emit(Inst{Result: statePtr, Op: OpStackAlloc, Imm: stateBytes})

		// Initialize state from initial value
		if stateType.IsTensor() {
			// Copy tensor data
			totalElements := 1
			for _, dim := range stateType.Shape {
				totalElements *= int(dim.Val)
			}
			for i := 0; i < totalElements; i++ {
				offset := int64(i * 4)
				elem := c.fresh(stateType.ElemType())
				c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{initState}, Imm: offset})
				c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, elem}, Imm: offset})
			}
		} else {
			// Store scalar
			c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, initState}, Imm: 0})
		}

		// Loop counter
		counterPtr := c.fresh(types.I64())
		c.emit(Inst{Result: counterPtr, Op: OpStackAlloc, Imm: 8})
		zero := c.fresh(types.I64())
		c.emit(Inst{Result: zero, Op: OpConstInt, Imm: 0})
		c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, zero}, Imm: 0})

		// Pre-allocate the increment constant BEFORE the loop
		// to avoid register conflicts with loop body values
		incOne := c.fresh(types.I64())
		c.emit(Inst{Result: incOne, Op: OpConstInt, Imm: 1})

		// Labels
		loopLabel := c.freshLabel()
		endLabel := c.freshLabel()

		// Loop structure:
		// loopLabel:
		//   counter = load(counterPtr)
		//   if counter < count: goto bodyLabel
		//   goto endLabel
		// bodyLabel:
		//   <body>
		//   counter++
		//   store(counterPtr, counter)
		//   goto loopLabel
		// endLabel:
		//   result = load(statePtr)

		bodyLabel := c.freshLabel()

		// Loop start
		c.emit(Inst{Op: OpLabel, LabelID: loopLabel})

		// Load counter
		counter := c.fresh(types.I64())
		c.emit(Inst{Result: counter, Op: OpLoad64, Args: []Value{counterPtr}, Imm: 0})

		// Check counter < count
		countVal := c.fresh(types.I64())
		c.emit(Inst{Result: countVal, Op: OpConstInt, Imm: int64(count)})
		ltCond := c.fresh(types.Bool())
		c.emit(Inst{Result: ltCond, Op: OpCmpLt, Args: []Value{counter, countVal}})

		// Jump to body if counter < count, else fall through to end
		c.emit(Inst{Op: OpJumpIf, Args: []Value{ltCond}, LabelID: bodyLabel})
		c.emit(Inst{Op: OpJump, LabelID: endLabel})

		// Identify captured tensor pointers that need to survive across iterations
		// We need to save them to stack slots before the loop and reload inside
		type capturedPtr struct {
			ceirID   int   // original CEIR value ID
			mirVal   Value // original MIR value (from valMap)
			saveSlot Value // stack slot to save the pointer
		}
		var capturedPtrs []capturedPtr

		// Find which values from valMap are used in the body instructions
		usedIDs := make(map[int]bool)
		for _, bodyInst := range in.BodyInsts {
			for _, arg := range bodyInst.Args {
				usedIDs[arg.ID] = true
			}
		}

		// For each used value that's in valMap (captured) and is a tensor/pointer, save it
		for ceirID := range usedIDs {
			if ceirID == in.LoopInID {
				continue // loop input handled separately
			}
			if mirVal, ok := valMap[ceirID]; ok {
				// This is a captured value - allocate a save slot and store it
				saveSlot := c.fresh(types.I64())
				c.emit(Inst{Result: saveSlot, Op: OpStackAlloc, Imm: 8})
				c.emit(Inst{Op: OpStore64, Args: []Value{saveSlot, mirVal}, Imm: 0})
				capturedPtrs = append(capturedPtrs, capturedPtr{
					ceirID:   ceirID,
					mirVal:   mirVal,
					saveSlot: saveSlot,
				})
			}
		}

		// Body label
		c.emit(Inst{Op: OpLabel, LabelID: bodyLabel})

		// Reload captured pointers at the start of each iteration
		bodyValMap := make(map[int]Value)
		for _, cap := range capturedPtrs {
			reloaded := c.fresh(types.I64())
			c.emit(Inst{Result: reloaded, Op: OpLoad64, Args: []Value{cap.saveSlot}, Imm: 0})
			bodyValMap[cap.ceirID] = reloaded
		}

		// For any values not captured as pointers, copy directly
		for k, v := range valMap {
			if _, ok := bodyValMap[k]; !ok {
				bodyValMap[k] = v
			}
		}

		// Load current state for loop body
		var loopInVal Value
		if stateType.IsTensor() {
			// State is a tensor pointer - the body will use it directly
			loopInVal = statePtr
		} else {
			// State is a scalar - load it
			loopInVal = c.fresh(stateType)
			c.emit(Inst{Result: loopInVal, Op: OpLoad32, Args: []Value{statePtr}, Imm: 0})
		}

		bodyValMap[in.LoopInID] = loopInVal

		// Process body instructions
		for _, bodyInst := range in.BodyInsts {
			c.lowerBodyInst(bodyInst, bodyValMap, stateType)
		}

		// Get the loop output value
		loopOutVal := bodyValMap[in.LoopOutID]

		// Store the new state
		if stateType.IsTensor() {
			// Copy tensor result back to statePtr
			totalElements := 1
			for _, dim := range stateType.Shape {
				totalElements *= int(dim.Val)
			}
			for i := 0; i < totalElements; i++ {
				offset := int64(i * 4)
				elem := c.fresh(stateType.ElemType())
				c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{loopOutVal}, Imm: offset})
				c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, elem}, Imm: offset})
			}
		} else {
			// Store scalar
			c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, loopOutVal}, Imm: 0})
		}

		// Increment counter (using pre-allocated incOne to avoid register conflicts)
		newCounter := c.fresh(types.I64())
		c.emit(Inst{Result: newCounter, Op: OpAdd, Args: []Value{counter, incOne}})
		c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, newCounter}, Imm: 0})

		// Jump back to loop
		c.emit(Inst{Op: OpJump, LabelID: loopLabel})

		// End label
		c.emit(Inst{Op: OpLabel, LabelID: endLabel})

		// Load final state as result
		if stateType.IsTensor() {
			valMap[in.Result.ID] = statePtr
		} else {
			finalResult := c.fresh(stateType)
			c.emit(Inst{Result: finalResult, Op: OpLoad32, Args: []Value{statePtr}, Imm: 0})
			valMap[in.Result.ID] = finalResult
		}

	case ceir.OpIterate:
		// Named function iterate with large count - emit loop
		initState := valMap[in.Args[0].ID]
		count := int(in.Imm)
		fnName := in.FnName
		stateType := in.Result.Type

		// Allocate state storage
		stateBytes := int64(8) // scalar for now
		if stateType.IsTensor() {
			totalElements := 1
			for _, dim := range stateType.Shape {
				totalElements *= int(dim.Val)
			}
			stateBytes = int64(totalElements * 4)
		}

		statePtr := c.fresh(types.I64())
		c.emit(Inst{Result: statePtr, Op: OpStackAlloc, Imm: stateBytes})

		// Initialize state
		if stateType.IsTensor() {
			totalElements := 1
			for _, dim := range stateType.Shape {
				totalElements *= int(dim.Val)
			}
			for i := 0; i < totalElements; i++ {
				offset := int64(i * 4)
				elem := c.fresh(stateType.ElemType())
				c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{initState}, Imm: offset})
				c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, elem}, Imm: offset})
			}
		} else {
			c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, initState}, Imm: 0})
		}

		// Loop counter
		counterPtr := c.fresh(types.I64())
		c.emit(Inst{Result: counterPtr, Op: OpStackAlloc, Imm: 8})
		zero := c.fresh(types.I64())
		c.emit(Inst{Result: zero, Op: OpConstInt, Imm: 0})
		c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, zero}, Imm: 0})

		// Labels
		loopLabel := c.freshLabel()
		bodyLabel := c.freshLabel()
		endLabel := c.freshLabel()

		// Pre-allocate the increment constant BEFORE the loop
		// to avoid register conflicts with loop body values
		incOne := c.fresh(types.I64())
		c.emit(Inst{Result: incOne, Op: OpConstInt, Imm: 1})

		// Loop start
		c.emit(Inst{Op: OpLabel, LabelID: loopLabel})

		// Load counter and compare
		counter := c.fresh(types.I64())
		c.emit(Inst{Result: counter, Op: OpLoad64, Args: []Value{counterPtr}, Imm: 0})
		countVal := c.fresh(types.I64())
		c.emit(Inst{Result: countVal, Op: OpConstInt, Imm: int64(count)})
		ltCond := c.fresh(types.Bool())
		c.emit(Inst{Result: ltCond, Op: OpCmpLt, Args: []Value{counter, countVal}})
		c.emit(Inst{Op: OpJumpIf, Args: []Value{ltCond}, LabelID: bodyLabel})
		c.emit(Inst{Op: OpJump, LabelID: endLabel})

		// Body
		c.emit(Inst{Op: OpLabel, LabelID: bodyLabel})

		// Load current state
		currentState := c.fresh(stateType)
		if stateType.IsTensor() {
			currentState = statePtr
		} else {
			c.emit(Inst{Result: currentState, Op: OpLoad32, Args: []Value{statePtr}, Imm: 0})
		}

		// Call function
		newState := c.fresh(stateType)
		c.emit(Inst{Result: newState, Op: OpCall, FnName: fnName, Args: []Value{currentState}})

		// Store new state
		if stateType.IsTensor() {
			totalElements := 1
			for _, dim := range stateType.Shape {
				totalElements *= int(dim.Val)
			}
			for i := 0; i < totalElements; i++ {
				offset := int64(i * 4)
				elem := c.fresh(stateType.ElemType())
				c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{newState}, Imm: offset})
				c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, elem}, Imm: offset})
			}
		} else {
			c.emit(Inst{Op: OpStore32, Args: []Value{statePtr, newState}, Imm: 0})
		}

		// Increment counter (use pre-allocated incOne)
		newCounter := c.fresh(types.I64())
		c.emit(Inst{Result: newCounter, Op: OpAdd, Args: []Value{counter, incOne}})
		c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, newCounter}, Imm: 0})

		// Jump back
		c.emit(Inst{Op: OpJump, LabelID: loopLabel})

		// End
		c.emit(Inst{Op: OpLabel, LabelID: endLabel})

		// Load final result
		if stateType.IsTensor() {
			valMap[in.Result.ID] = statePtr
		} else {
			finalResult := c.fresh(stateType)
			c.emit(Inst{Result: finalResult, Op: OpLoad32, Args: []Value{statePtr}, Imm: 0})
			valMap[in.Result.ID] = finalResult
		}

	case ceir.OpTabulateLoop:
		c.lowerTabulateLoop(in, valMap)

	case ceir.OpScanLoop:
		c.lowerScanLoop(in, valMap)

	case ceir.OpEachLoop:
		c.lowerEachLoop(in, valMap)

	case ceir.OpConstString:
		// Intern the literal bytes in .rodata under a fresh symbol,
		// then represent the value as the rodata pointer (an i64).
		// Length is tracked in c.stringLengths so print_str can
		// recover it without dragging an extra SSA Value through
		// regalloc.
		if c.mod == nil {
			ptr := c.fresh(types.I64())
			c.emit(Inst{Result: ptr, Op: OpConstInt, Imm: 0})
			c.stringLengths[in.Result.ID] = in.Imm
			valMap[in.Result.ID] = ptr
			return
		}
		sym := fmt.Sprintf("__rodata_S%d", len(c.mod.Rodata))
		c.mod.Rodata = append(c.mod.Rodata, RodataEntry{
			Sym:   sym,
			Bytes: []byte(in.ImmS),
			Align: 1,
		})
		ptr := c.fresh(types.I64())
		c.emit(Inst{Result: ptr, Op: OpRodataPtr, FnName: sym})
		c.stringLengths[in.Result.ID] = in.Imm
		valMap[in.Result.ID] = ptr

	case ceir.OpCall:
		// Special-case print_str: expand the single string-typed CEIR
		// arg into a two-arg (ptr, len) System V call. The pointer
		// comes from valMap (set by OpConstString lowering); the
		// length is in c.stringLengths.
		if in.FnName == "print_str" && len(in.Args) == 1 {
			strArg := in.Args[0]
			ptr := valMap[strArg.ID]
			length := c.stringLengths[strArg.ID]
			lenVal := c.fresh(types.I64())
			c.emit(Inst{Result: lenVal, Op: OpConstInt, Imm: length})
			result := c.fresh(in.Result.Type)
			c.emit(Inst{
				Result: result,
				Op:     OpCall,
				FnName: in.FnName,
				Args:   []Value{ptr, lenVal},
			})
			// print_str returns the same fat pointer; propagate the
			// length so chained uses (none in v0.12, but symmetric)
			// work without re-deriving it.
			c.stringLengths[in.Result.ID] = length
			valMap[in.Result.ID] = result
			return
		}
		// Generic call: fall through to the default scalar path.
		args := make([]Value, len(in.Args))
		for i, a := range in.Args {
			args[i] = valMap[a.ID]
		}
		result := c.fresh(in.Result.Type)
		c.emit(Inst{
			Result: result,
			Op:     OpCall,
			Imm:    in.Imm,
			ImmF:   in.ImmF,
			ImmB:   in.ImmB,
			FnName: in.FnName,
			Args:   args,
		})
		valMap[in.Result.ID] = result

	default:
		// Simple translation for scalar ops
		args := make([]Value, len(in.Args))
		for i, a := range in.Args {
			args[i] = valMap[a.ID]
		}
		result := c.fresh(in.Result.Type)
		c.emit(Inst{
			Result: result,
			Op:     translateOp(in.Op),
			Imm:    in.Imm,
			ImmF:   in.ImmF,
			ImmB:   in.ImmB,
			FnName: in.FnName,
			Args:   args,
		})
		valMap[in.Result.ID] = result
	}
}

// loadOpForElem returns the GPR or SSE load opcode appropriate for
// the given tensor element type. i32 elements load through GPRs;
// f32 elements load into SSE registers.
func loadOpForElem(elem *types.Type) Op {
	if elem != nil && elem.K == types.KI32 {
		return OpLoadI32
	}
	return OpLoad32
}

// storeOpForElem mirrors loadOpForElem on the store side.
func storeOpForElem(elem *types.Type) Op {
	if elem != nil && elem.K == types.KI32 {
		return OpStoreI32
	}
	return OpStore32
}

// captureOuterPtrs identifies outer-context values (CEIR IDs already
// in valMap) that the loop body refers to, saves each one to a fresh
// stack slot before the loop, and returns a list describing the slots
// so the body label can reload them. Mirrors the OpIterateLoop pattern.
func (c *lowerCtx) captureOuterPtrs(bodyInsts []ceir.Inst, valMap map[int]Value, skipIDs map[int]bool) []capturedPtrSpec {
	used := make(map[int]bool)
	for _, b := range bodyInsts {
		for _, a := range b.Args {
			used[a.ID] = true
		}
	}
	var caps []capturedPtrSpec
	for id := range used {
		if skipIDs[id] {
			continue
		}
		mv, ok := valMap[id]
		if !ok {
			continue
		}
		slot := c.fresh(types.I64())
		c.emit(Inst{Result: slot, Op: OpStackAlloc, Imm: 8})
		c.emit(Inst{Op: OpStore64, Args: []Value{slot, mv}, Imm: 0})
		caps = append(caps, capturedPtrSpec{ceirID: id, slot: slot})
	}
	return caps
}

type capturedPtrSpec struct {
	ceirID int
	slot   Value
}

// lowerTabulateLoop emits a counted runtime loop that fills an output
// tensor of length Imm. Each iteration binds the loop counter as the
// CEIR LoopInID, runs BodyInsts, then stores LoopOutID into out[i*4].
func (c *lowerCtx) lowerTabulateLoop(in ceir.Inst, valMap map[int]Value) {
	count := in.Imm
	resultType := in.Result.Type
	elemType := resultType.ElemType()

	bytes := count * 4
	outBase := c.fresh(types.I64())
	c.emit(Inst{Result: outBase, Op: OpStackAlloc, Imm: bytes})

	counterPtr := c.fresh(types.I64())
	c.emit(Inst{Result: counterPtr, Op: OpStackAlloc, Imm: 8})
	zero := c.fresh(types.I64())
	c.emit(Inst{Result: zero, Op: OpConstInt, Imm: 0})
	c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, zero}, Imm: 0})

	loopLabel := c.freshLabel()
	bodyLabel := c.freshLabel()
	endLabel := c.freshLabel()

	caps := c.captureOuterPtrs(in.BodyInsts, valMap, map[int]bool{in.LoopInID: true})

	// loopLabel:
	c.emit(Inst{Op: OpLabel, LabelID: loopLabel})

	counter := c.fresh(types.I64())
	c.emit(Inst{Result: counter, Op: OpLoad64, Args: []Value{counterPtr}, Imm: 0})
	countVal := c.fresh(types.I64())
	c.emit(Inst{Result: countVal, Op: OpConstInt, Imm: count})
	cmp := c.fresh(types.Bool())
	c.emit(Inst{Result: cmp, Op: OpCmpLt, Args: []Value{counter, countVal}})
	c.emit(Inst{Op: OpJumpIf, Args: []Value{cmp}, LabelID: bodyLabel})
	c.emit(Inst{Op: OpJump, LabelID: endLabel})

	// bodyLabel:
	c.emit(Inst{Op: OpLabel, LabelID: bodyLabel})

	// Re-emit the loop-local constants inside the body. The linear-scan
	// regalloc treats their last MIR use as terminating their live
	// interval, so a definition above the back-edge would let a later
	// MIR inst reuse the same physical register and corrupt the value
	// for the next iteration. Re-emitting at bodyLabel makes each
	// runtime iteration's `mov rN, K` re-establish the constant.
	incOne := c.fresh(types.I64())
	c.emit(Inst{Result: incOne, Op: OpConstInt, Imm: 1})
	four := c.fresh(types.I64())
	c.emit(Inst{Result: four, Op: OpConstInt, Imm: 4})

	bodyValMap := make(map[int]Value)
	for _, cap := range caps {
		reload := c.fresh(types.I64())
		c.emit(Inst{Result: reload, Op: OpLoad64, Args: []Value{cap.slot}, Imm: 0})
		bodyValMap[cap.ceirID] = reload
	}
	for k, v := range valMap {
		if _, ok := bodyValMap[k]; !ok {
			bodyValMap[k] = v
		}
	}

	// The CEIR loop input is the iteration index (i32). Truncate the
	// 64-bit counter via OpCopy — the regalloc treats i32 and i64 as
	// the same GPR width on the in-register path.
	idx := c.fresh(types.I32())
	c.emit(Inst{Result: idx, Op: OpCopy, Args: []Value{counter}})
	bodyValMap[in.LoopInID] = idx

	for _, bi := range in.BodyInsts {
		c.lowerBodyInst(bi, bodyValMap, resultType)
	}

	elemVal := bodyValMap[in.LoopOutID]

	// out[counter*4] = elemVal
	off := c.fresh(types.I64())
	c.emit(Inst{Result: off, Op: OpMul, Args: []Value{counter, four}})
	addr := c.fresh(types.I64())
	c.emit(Inst{Result: addr, Op: OpAdd, Args: []Value{outBase, off}})
	c.emit(Inst{Op: storeOpForElem(elemType), Args: []Value{addr, elemVal}, Imm: 0})

	// counter++
	newCounter := c.fresh(types.I64())
	c.emit(Inst{Result: newCounter, Op: OpAdd, Args: []Value{counter, incOne}})
	c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, newCounter}, Imm: 0})
	c.emit(Inst{Op: OpJump, LabelID: loopLabel})

	// endLabel:
	c.emit(Inst{Op: OpLabel, LabelID: endLabel})
	valMap[in.Result.ID] = outBase
}

// lowerScanLoop emits a counted runtime loop. Args[0] is the initial
// accumulator, Args[1] the input tensor base. Each iteration loads
// elem = in[i*4], runs BodyInsts with (acc, elem) bound, and stores
// the new acc into both the rolling acc slot and out[i*4].
func (c *lowerCtx) lowerScanLoop(in ceir.Inst, valMap map[int]Value) {
	count := in.Imm
	resultType := in.Result.Type
	elemType := resultType.ElemType()

	initAcc := valMap[in.Args[0].ID]
	inputBase := valMap[in.Args[1].ID]

	bytes := count * 4
	outBase := c.fresh(types.I64())
	c.emit(Inst{Result: outBase, Op: OpStackAlloc, Imm: bytes})

	// Acc lives in a stack slot so it threads across iterations.
	accSlot := c.fresh(types.I64())
	c.emit(Inst{Result: accSlot, Op: OpStackAlloc, Imm: 8})
	c.emit(Inst{Op: storeOpForElem(elemType), Args: []Value{accSlot, initAcc}, Imm: 0})

	counterPtr := c.fresh(types.I64())
	c.emit(Inst{Result: counterPtr, Op: OpStackAlloc, Imm: 8})
	zero := c.fresh(types.I64())
	c.emit(Inst{Result: zero, Op: OpConstInt, Imm: 0})
	c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, zero}, Imm: 0})

	// Save the input tensor base in a stack slot so we can reload it
	// inside the loop without depending on regalloc keeping it live
	// across the body. The same trick OpIterateLoop uses for captures.
	inputSlot := c.fresh(types.I64())
	c.emit(Inst{Result: inputSlot, Op: OpStackAlloc, Imm: 8})
	c.emit(Inst{Op: OpStore64, Args: []Value{inputSlot, inputBase}, Imm: 0})

	loopLabel := c.freshLabel()
	bodyLabel := c.freshLabel()
	endLabel := c.freshLabel()

	caps := c.captureOuterPtrs(in.BodyInsts, valMap, map[int]bool{
		in.LoopInID:  true,
		in.LoopInID2: true,
	})

	c.emit(Inst{Op: OpLabel, LabelID: loopLabel})
	counter := c.fresh(types.I64())
	c.emit(Inst{Result: counter, Op: OpLoad64, Args: []Value{counterPtr}, Imm: 0})
	countVal := c.fresh(types.I64())
	c.emit(Inst{Result: countVal, Op: OpConstInt, Imm: count})
	cmp := c.fresh(types.Bool())
	c.emit(Inst{Result: cmp, Op: OpCmpLt, Args: []Value{counter, countVal}})
	c.emit(Inst{Op: OpJumpIf, Args: []Value{cmp}, LabelID: bodyLabel})
	c.emit(Inst{Op: OpJump, LabelID: endLabel})

	c.emit(Inst{Op: OpLabel, LabelID: bodyLabel})

	// Re-emit loop-local constants inside the body so they survive the
	// back-edge. See lowerTabulateLoop for the full rationale.
	incOne := c.fresh(types.I64())
	c.emit(Inst{Result: incOne, Op: OpConstInt, Imm: 1})
	four := c.fresh(types.I64())
	c.emit(Inst{Result: four, Op: OpConstInt, Imm: 4})

	bodyValMap := make(map[int]Value)
	for _, cap := range caps {
		reload := c.fresh(types.I64())
		c.emit(Inst{Result: reload, Op: OpLoad64, Args: []Value{cap.slot}, Imm: 0})
		bodyValMap[cap.ceirID] = reload
	}
	for k, v := range valMap {
		if _, ok := bodyValMap[k]; !ok {
			bodyValMap[k] = v
		}
	}

	// Reload acc from its slot.
	acc := c.fresh(elemType)
	c.emit(Inst{Result: acc, Op: loadOpForElem(elemType), Args: []Value{accSlot}, Imm: 0})
	bodyValMap[in.LoopInID] = acc

	// Reload input tensor base, compute elem = in[counter*4].
	inBase := c.fresh(types.I64())
	c.emit(Inst{Result: inBase, Op: OpLoad64, Args: []Value{inputSlot}, Imm: 0})
	off := c.fresh(types.I64())
	c.emit(Inst{Result: off, Op: OpMul, Args: []Value{counter, four}})
	inAddr := c.fresh(types.I64())
	c.emit(Inst{Result: inAddr, Op: OpAdd, Args: []Value{inBase, off}})
	elem := c.fresh(elemType)
	c.emit(Inst{Result: elem, Op: loadOpForElem(elemType), Args: []Value{inAddr}, Imm: 0})
	bodyValMap[in.LoopInID2] = elem

	for _, bi := range in.BodyInsts {
		c.lowerBodyInst(bi, bodyValMap, resultType)
	}

	newAcc := bodyValMap[in.LoopOutID]

	// Store acc back into rolling slot AND into out[counter*4]. outBase
	// has uses after the loop (the result feeds whatever consumer reads
	// the scan output), so the linear-scan regalloc naturally extends
	// its live interval past the loop body and keeps it in a register.
	c.emit(Inst{Op: storeOpForElem(elemType), Args: []Value{accSlot, newAcc}, Imm: 0})
	// Recompute the byte offset so the original `off` from the input-load
	// path doesn't have to live across the body. Re-emit `four` too so
	// the constant has a tight live range bounded by its second use.
	four2 := c.fresh(types.I64())
	c.emit(Inst{Result: four2, Op: OpConstInt, Imm: 4})
	off2 := c.fresh(types.I64())
	c.emit(Inst{Result: off2, Op: OpMul, Args: []Value{counter, four2}})
	outAddr := c.fresh(types.I64())
	c.emit(Inst{Result: outAddr, Op: OpAdd, Args: []Value{outBase, off2}})
	c.emit(Inst{Op: storeOpForElem(elemType), Args: []Value{outAddr, newAcc}, Imm: 0})

	// counter++
	newCounter := c.fresh(types.I64())
	c.emit(Inst{Result: newCounter, Op: OpAdd, Args: []Value{counter, incOne}})
	c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, newCounter}, Imm: 0})
	c.emit(Inst{Op: OpJump, LabelID: loopLabel})

	c.emit(Inst{Op: OpLabel, LabelID: endLabel})
	valMap[in.Result.ID] = outBase
}

// lowerEachLoop emits a counted runtime loop that walks Args[0]
// (rank-1 tensor of length Imm) and calls FnName(elem) per element.
// FnName is whatever callee the type checker accepted (any function
// whose effects fit the caller's effect set in v0.13); the result
// (unit) is unused.
func (c *lowerCtx) lowerEachLoop(in ceir.Inst, valMap map[int]Value) {
	count := in.Imm
	tensorBase := valMap[in.Args[0].ID]
	elemType := in.Args[0].Type.ElemType()

	// Save the tensor base across the call. Without this the
	// pointer can be clobbered by a caller-saved register inside the
	// callee.
	baseSlot := c.fresh(types.I64())
	c.emit(Inst{Result: baseSlot, Op: OpStackAlloc, Imm: 8})
	c.emit(Inst{Op: OpStore64, Args: []Value{baseSlot, tensorBase}, Imm: 0})

	counterPtr := c.fresh(types.I64())
	c.emit(Inst{Result: counterPtr, Op: OpStackAlloc, Imm: 8})
	zero := c.fresh(types.I64())
	c.emit(Inst{Result: zero, Op: OpConstInt, Imm: 0})
	c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, zero}, Imm: 0})

	loopLabel := c.freshLabel()
	bodyLabel := c.freshLabel()
	endLabel := c.freshLabel()

	c.emit(Inst{Op: OpLabel, LabelID: loopLabel})
	counter := c.fresh(types.I64())
	c.emit(Inst{Result: counter, Op: OpLoad64, Args: []Value{counterPtr}, Imm: 0})
	countVal := c.fresh(types.I64())
	c.emit(Inst{Result: countVal, Op: OpConstInt, Imm: count})
	cmp := c.fresh(types.Bool())
	c.emit(Inst{Result: cmp, Op: OpCmpLt, Args: []Value{counter, countVal}})
	c.emit(Inst{Op: OpJumpIf, Args: []Value{cmp}, LabelID: bodyLabel})
	c.emit(Inst{Op: OpJump, LabelID: endLabel})

	c.emit(Inst{Op: OpLabel, LabelID: bodyLabel})

	// Re-emit loop-local constants inside the body. See lowerTabulateLoop.
	incOne := c.fresh(types.I64())
	c.emit(Inst{Result: incOne, Op: OpConstInt, Imm: 1})
	four := c.fresh(types.I64())
	c.emit(Inst{Result: four, Op: OpConstInt, Imm: 4})

	// Reload base, load elem, call.
	base := c.fresh(types.I64())
	c.emit(Inst{Result: base, Op: OpLoad64, Args: []Value{baseSlot}, Imm: 0})
	off := c.fresh(types.I64())
	c.emit(Inst{Result: off, Op: OpMul, Args: []Value{counter, four}})
	addr := c.fresh(types.I64())
	c.emit(Inst{Result: addr, Op: OpAdd, Args: []Value{base, off}})
	elem := c.fresh(elemType)
	c.emit(Inst{Result: elem, Op: loadOpForElem(elemType), Args: []Value{addr}, Imm: 0})

	callRes := c.fresh(types.Unit())
	c.emit(Inst{Result: callRes, Op: OpCall, FnName: in.FnName, Args: []Value{elem}})

	// Reload counter (the call may have clobbered it) and increment.
	counterAfter := c.fresh(types.I64())
	c.emit(Inst{Result: counterAfter, Op: OpLoad64, Args: []Value{counterPtr}, Imm: 0})
	newCounter := c.fresh(types.I64())
	c.emit(Inst{Result: newCounter, Op: OpAdd, Args: []Value{counterAfter, incOne}})
	c.emit(Inst{Op: OpStore64, Args: []Value{counterPtr, newCounter}, Imm: 0})
	c.emit(Inst{Op: OpJump, LabelID: loopLabel})

	c.emit(Inst{Op: OpLabel, LabelID: endLabel})

	// each returns unit.
	unitVal := c.fresh(types.Unit())
	c.emit(Inst{Result: unitVal, Op: OpConstInt, Imm: 0})
	valMap[in.Result.ID] = unitVal
}

// lowerBodyInst lowers a CEIR instruction from a loop body into MIR
func (c *lowerCtx) lowerBodyInst(in ceir.Inst, valMap map[int]Value, stateType *types.Type) {
	// Most body instructions are simple ops that just need value mapping
	switch in.Op {
	case ceir.OpTensorAdd, ceir.OpTensorSub, ceir.OpTensorMul, ceir.OpTensorDiv:
		// Element-wise tensor operation
		aBase := valMap[in.Args[0].ID]
		bBase := valMap[in.Args[1].ID]

		tensorType := in.Result.Type
		totalElements := 1
		for _, dim := range tensorType.Shape {
			totalElements *= int(dim.Val)
		}

		// Allocate result
		bytes := int64(totalElements * 4)
		resultBase := c.fresh(types.I64())
		c.emit(Inst{Result: resultBase, Op: OpStackAlloc, Imm: bytes})

		// Determine operations
		var floatOp, packedOp128, packedOp256 Op
		switch in.Op {
		case ceir.OpTensorAdd:
			floatOp = OpAddF32
			packedOp128 = OpAddPS128
			packedOp256 = OpAddPS256
		case ceir.OpTensorSub:
			floatOp = OpSubF32
			packedOp128 = OpSubPS128
			packedOp256 = OpSubPS256
		case ceir.OpTensorMul:
			floatOp = OpMulF32
			packedOp128 = OpMulPS128
			packedOp256 = OpMulPS256
		case ceir.OpTensorDiv:
			floatOp = OpDivF32
			packedOp128 = OpDivPS128
			packedOp256 = OpDivPS256
		}

		elemType := tensorType.ElemType()
		remaining := totalElements
		offset := int64(0)

		// AVX2: 8 elements at a time
		for remaining >= 8 {
			aVec := c.fresh(types.Vec8F32())
			c.emit(Inst{Result: aVec, Op: OpLoad256, Args: []Value{aBase}, Imm: offset})
			bVec := c.fresh(types.Vec8F32())
			c.emit(Inst{Result: bVec, Op: OpLoad256, Args: []Value{bBase}, Imm: offset})
			rVec := c.fresh(types.Vec8F32())
			c.emit(Inst{Result: rVec, Op: packedOp256, Args: []Value{aVec, bVec}})
			c.emit(Inst{Op: OpStore256, Args: []Value{resultBase, rVec}, Imm: offset})
			offset += 32
			remaining -= 8
		}

		// SSE: 4 elements at a time
		for remaining >= 4 {
			aVec := c.fresh(types.Vec4F32())
			c.emit(Inst{Result: aVec, Op: OpLoad128, Args: []Value{aBase}, Imm: offset})
			bVec := c.fresh(types.Vec4F32())
			c.emit(Inst{Result: bVec, Op: OpLoad128, Args: []Value{bBase}, Imm: offset})
			rVec := c.fresh(types.Vec4F32())
			c.emit(Inst{Result: rVec, Op: packedOp128, Args: []Value{aVec, bVec}})
			c.emit(Inst{Op: OpStore128, Args: []Value{resultBase, rVec}, Imm: offset})
			offset += 16
			remaining -= 4
		}

		// Scalar cleanup
		for remaining > 0 {
			aElem := c.fresh(elemType)
			c.emit(Inst{Result: aElem, Op: OpLoad32, Args: []Value{aBase}, Imm: offset})
			bElem := c.fresh(elemType)
			c.emit(Inst{Result: bElem, Op: OpLoad32, Args: []Value{bBase}, Imm: offset})
			rElem := c.fresh(elemType)
			c.emit(Inst{Result: rElem, Op: floatOp, Args: []Value{aElem, bElem}})
			c.emit(Inst{Op: OpStore32, Args: []Value{resultBase, rElem}, Imm: offset})
			offset += 4
			remaining--
		}

		valMap[in.Result.ID] = resultBase

	case ceir.OpTensorLit:
		// Handle tensor literal in body
		tensorType := in.Result.Type
		totalElements := 1
		for _, dim := range tensorType.Shape {
			totalElements *= int(dim.Val)
		}
		bytes := int64(totalElements * 4)

		baseVal := c.fresh(types.I64())
		c.emit(Inst{Result: baseVal, Op: OpStackAlloc, Imm: bytes})

		if len(in.Args) > 0 && in.Args[0].Type.IsTensor() {
			innerElements := 1
			for i := 1; i < len(tensorType.Shape); i++ {
				innerElements *= int(tensorType.Shape[i].Val)
			}
			for i, arg := range in.Args {
				srcBase := valMap[arg.ID]
				dstOffset := int64(i * innerElements * 4)
				for j := 0; j < innerElements; j++ {
					elemOffset := int64(j * 4)
					elem := c.fresh(tensorType.ElemType())
					c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{srcBase}, Imm: elemOffset})
					c.emit(Inst{Op: OpStore32, Args: []Value{baseVal, elem}, Imm: dstOffset + elemOffset})
				}
			}
		} else {
			for i, arg := range in.Args {
				srcVal := valMap[arg.ID]
				offset := int64(i * 4)
				c.emit(Inst{Op: OpStore32, Args: []Value{baseVal, srcVal}, Imm: offset})
			}
		}
		valMap[in.Result.ID] = baseVal

	default:
		// Simple scalar ops
		args := make([]Value, len(in.Args))
		for i, a := range in.Args {
			args[i] = valMap[a.ID]
		}
		result := c.fresh(in.Result.Type)
		c.emit(Inst{
			Result: result,
			Op:     translateOp(in.Op),
			Imm:    in.Imm,
			ImmF:   in.ImmF,
			ImmB:   in.ImmB,
			FnName: in.FnName,
			Args:   args,
		})
		valMap[in.Result.ID] = result
	}
}

func (c *lowerCtx) emitElementWiseLoop(aBase, bBase, resultBase Value, n int, op Op, elemType *types.Type) {
	// Loop variable i
	i := c.fresh(types.I64())
	c.emit(Inst{Result: i, Op: OpConstInt, Imm: 0})

	loopLabel := c.freshLabel()
	endLabel := c.freshLabel()

	// loop:
	c.emit(Inst{Op: OpLabel, LabelID: loopLabel})

	// Load a[i] and b[i]
	// Compute offset = i * 4
	four := c.fresh(types.I64())
	c.emit(Inst{Result: four, Op: OpConstInt, Imm: 4})
	offset := c.fresh(types.I64())
	c.emit(Inst{Result: offset, Op: OpMul, Args: []Value{i, four}})

	// Calculate addresses and load
	aAddr := c.fresh(types.I64())
	c.emit(Inst{Result: aAddr, Op: OpAdd, Args: []Value{aBase, offset}})
	aElem := c.fresh(elemType)
	c.emit(Inst{Result: aElem, Op: OpLoad32, Args: []Value{aAddr}, Imm: 0})

	bAddr := c.fresh(types.I64())
	c.emit(Inst{Result: bAddr, Op: OpAdd, Args: []Value{bBase, offset}})
	bElem := c.fresh(elemType)
	c.emit(Inst{Result: bElem, Op: OpLoad32, Args: []Value{bAddr}, Imm: 0})

	// Compute result
	rElem := c.fresh(elemType)
	c.emit(Inst{Result: rElem, Op: op, Args: []Value{aElem, bElem}})

	// Store result[i]
	rAddr := c.fresh(types.I64())
	c.emit(Inst{Result: rAddr, Op: OpAdd, Args: []Value{resultBase, offset}})
	c.emit(Inst{Op: OpStore32, Args: []Value{rAddr, rElem}, Imm: 0})

	// i = i + 1
	one := c.fresh(types.I64())
	c.emit(Inst{Result: one, Op: OpConstInt, Imm: 1})
	newI := c.fresh(types.I64())
	c.emit(Inst{Result: newI, Op: OpAdd, Args: []Value{i, one}})

	// if i < n: goto loop
	nVal := c.fresh(types.I64())
	c.emit(Inst{Result: nVal, Op: OpConstInt, Imm: int64(n)})
	cmp := c.fresh(types.Bool())
	c.emit(Inst{Result: cmp, Op: OpCmpLt, Args: []Value{newI, nVal}})
	c.emit(Inst{Op: OpJumpIf, Args: []Value{cmp}, LabelID: loopLabel})

	// end:
	c.emit(Inst{Op: OpLabel, LabelID: endLabel})
}

func (c *lowerCtx) emitReductionLoop(tensorBase, acc Value, n int, op Op, elemType *types.Type) Value {
	// Loop variable i
	i := c.fresh(types.I64())
	c.emit(Inst{Result: i, Op: OpConstInt, Imm: 0})

	loopLabel := c.freshLabel()
	endLabel := c.freshLabel()

	// We need phi-like behavior for acc. For now, use memory.
	// Actually, let's just unroll to keep it simple for v0.3

	// loop:
	c.emit(Inst{Op: OpLabel, LabelID: loopLabel})

	// Load element at offset i*4
	four := c.fresh(types.I64())
	c.emit(Inst{Result: four, Op: OpConstInt, Imm: 4})
	offset := c.fresh(types.I64())
	c.emit(Inst{Result: offset, Op: OpMul, Args: []Value{i, four}})

	addr := c.fresh(types.I64())
	c.emit(Inst{Result: addr, Op: OpAdd, Args: []Value{tensorBase, offset}})
	elem := c.fresh(elemType)
	c.emit(Inst{Result: elem, Op: OpLoad32, Args: []Value{addr}, Imm: 0})

	// acc = acc op elem
	newAcc := c.fresh(elemType)
	c.emit(Inst{Result: newAcc, Op: op, Args: []Value{acc, elem}})

	// i = i + 1
	one := c.fresh(types.I64())
	c.emit(Inst{Result: one, Op: OpConstInt, Imm: 1})
	newI := c.fresh(types.I64())
	c.emit(Inst{Result: newI, Op: OpAdd, Args: []Value{i, one}})

	// if i < n: goto loop
	nVal := c.fresh(types.I64())
	c.emit(Inst{Result: nVal, Op: OpConstInt, Imm: int64(n)})
	cmp := c.fresh(types.Bool())
	c.emit(Inst{Result: cmp, Op: OpCmpLt, Args: []Value{newI, nVal}})
	c.emit(Inst{Op: OpJumpIf, Args: []Value{cmp}, LabelID: loopLabel})

	// end:
	c.emit(Inst{Op: OpLabel, LabelID: endLabel})

	return newAcc
}

func (c *lowerCtx) emitMatMulLoops(aBase, bBase, cBase Value, M, K, N int, elemType *types.Type) {
	// Generate triple-nested loops for matmul
	// For now, just unroll everything (not ideal but works)
	// TODO: emit actual loop instructions for large matrices
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			acc := c.fresh(elemType)
			c.emit(Inst{Result: acc, Op: OpConstFloat, ImmF: 0.0})

			for k := 0; k < K; k++ {
				aOffset := int64((i*K + k) * 4)
				aElem := c.fresh(elemType)
				c.emit(Inst{Result: aElem, Op: OpLoad32, Args: []Value{aBase}, Imm: aOffset})

				bOffset := int64((k*N + j) * 4)
				bElem := c.fresh(elemType)
				c.emit(Inst{Result: bElem, Op: OpLoad32, Args: []Value{bBase}, Imm: bOffset})

				prod := c.fresh(elemType)
				c.emit(Inst{Result: prod, Op: OpMulF32, Args: []Value{aElem, bElem}})
				newAcc := c.fresh(elemType)
				c.emit(Inst{Result: newAcc, Op: OpAddF32, Args: []Value{acc, prod}})
				acc = newAcc
			}

			cOffset := int64((i*N + j) * 4)
			c.emit(Inst{Op: OpStore32, Args: []Value{cBase, acc}, Imm: cOffset})
		}
	}
}

func translateOp(o ceir.Op) Op {
	switch o {
	case ceir.OpConstInt:
		return OpConstInt
	case ceir.OpConstFloat:
		return OpConstFloat
	case ceir.OpConstBool:
		return OpConstBool
	case ceir.OpAdd:
		return OpAdd
	case ceir.OpSub:
		return OpSub
	case ceir.OpMul:
		return OpMul
	case ceir.OpDiv:
		return OpDiv
	case ceir.OpMod:
		return OpMod
	case ceir.OpNeg:
		return OpNeg
	case ceir.OpNot:
		return OpNot
	case ceir.OpEq:
		return OpEq
	case ceir.OpNe:
		return OpNe
	case ceir.OpLt:
		return OpLt
	case ceir.OpLe:
		return OpLe
	case ceir.OpGt:
		return OpGt
	case ceir.OpGe:
		return OpGe
	case ceir.OpAnd:
		return OpAnd
	case ceir.OpOr:
		return OpOr
	case ceir.OpCall:
		return OpCall
	case ceir.OpSelect:
		return OpSelect
	case ceir.OpCopy:
		return OpCopy
	// Cast operations
	case ceir.OpCastF32ToI32:
		return OpCastF32ToI32
	case ceir.OpCastF64ToI32:
		return OpCastF64ToI32
	case ceir.OpCastI32ToF32:
		return OpCastI32ToF32
	case ceir.OpCastI32ToF64:
		return OpCastI32ToF64
	case ceir.OpCastI8ToI32:
		return OpCastI8ToI32
	case ceir.OpCastI8ToI64:
		return OpCastI8ToI64
	case ceir.OpCastU8ToI32:
		return OpCastU8ToI32
	case ceir.OpCastU8ToI64:
		return OpCastU8ToI64
	case ceir.OpCastI32ToI8:
		return OpCastI32ToI8
	case ceir.OpCastI32ToU8:
		return OpCastI32ToU8
	case ceir.OpCastI64ToI8:
		return OpCastI64ToI8
	case ceir.OpCastI64ToU8:
		return OpCastI64ToU8
	// Tensor ops should not reach here (handled specially in lowerInst)
	case ceir.OpTensorLit, ceir.OpTensorAdd, ceir.OpTensorSub, ceir.OpTensorMul, ceir.OpTensorDiv, ceir.OpReduceSum, ceir.OpReduceProd, ceir.OpMatMul:
		return OpInvalid
	// Loop ops are handled specially
	case ceir.OpIterateLoop, ceir.OpIterate,
		ceir.OpTabulateLoop, ceir.OpScanLoop, ceir.OpEachLoop:
		return OpInvalid
	}
	return OpInvalid
}

// scanConstants walks the CEIR function in program order recording
// every value that has a compile-time-known representation:
//
//   - OpConstFloat / OpConstInt → scalar bytes in c.constScalars.
//   - OpTensorLit whose Args are all already-known scalars or already-
//     known constant tensors → flat little-endian bytes in
//     c.constTensors.
//
// The result drives the .rodata fast path in OpTensorLit lowering.
// Any tensor that depends on a non-constant input (parameter, call
// result, arithmetic) is simply absent from the map and falls back to
// the stack-store path.
func (c *lowerCtx) scanConstants(f *ceir.Func) {
	for _, in := range f.Body {
		switch in.Op {
		case ceir.OpConstFloat:
			// Tensor element type drives the encoding. The CEIR type
			// system uses f32 for tensor elements; honour that here.
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(float32(in.ImmF)))
			c.constScalars[in.Result.ID] = append([]byte(nil), b[:]...)

		case ceir.OpConstInt:
			// i32 packs into 4 bytes little-endian; tensors of i32 are
			// rare today but we support the encoding for symmetry.
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], uint32(int32(in.Imm)))
			c.constScalars[in.Result.ID] = append([]byte(nil), b[:]...)

		case ceir.OpTensorLit:
			blob, ok := c.flattenTensorLit(in)
			if ok {
				c.constTensors[in.Result.ID] = blob
			}
		}
	}
}

// computeAbsorbedConsts returns the set of CEIR scalar-constant value
// IDs whose every use is as an arg of a tensor literal that was
// promoted to .rodata (i.e. present in constTensors). Such constants
// have no surviving consumer in MIR and should be skipped during
// lowering.
func computeAbsorbedConsts(f *ceir.Func, constTensors map[int][]byte) map[int]bool {
	useCount := make(map[int]int)
	absorbedUses := make(map[int]int)
	for _, in := range f.Body {
		_, intoRodata := constTensors[in.Result.ID]
		isLit := in.Op == ceir.OpTensorLit
		for _, arg := range in.Args {
			useCount[arg.ID]++
			if isLit && intoRodata {
				absorbedUses[arg.ID]++
			}
		}
	}
	out := make(map[int]bool)
	for id, n := range useCount {
		if n > 0 && absorbedUses[id] == n {
			out[id] = true
		}
	}
	return out
}

// flattenTensorLit returns the raw little-endian byte payload for a
// fully-constant tensor literal, or (nil, false) if any leaf is not a
// known constant. Rank-2+ literals concatenate sub-tensor blobs in
// row-major order.
func (c *lowerCtx) flattenTensorLit(in ceir.Inst) ([]byte, bool) {
	tensorType := in.Result.Type
	totalElems := 1
	for _, d := range tensorType.Shape {
		totalElems *= int(d.Val)
	}
	out := make([]byte, 0, totalElems*4)

	if len(in.Args) > 0 && in.Args[0].Type.IsTensor() {
		// Rank-2+: every arg must already be a constant sub-tensor.
		for _, arg := range in.Args {
			sub, ok := c.constTensors[arg.ID]
			if !ok {
				return nil, false
			}
			out = append(out, sub...)
		}
		return out, true
	}

	// Rank-1: every arg must be a constant scalar.
	for _, arg := range in.Args {
		s, ok := c.constScalars[arg.ID]
		if !ok {
			return nil, false
		}
		out = append(out, s...)
	}
	return out, true
}
