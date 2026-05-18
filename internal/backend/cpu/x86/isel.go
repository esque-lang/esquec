package x86

import (
	"fmt"
	"math"

	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/types"
)

// CompiledFn is the per-function output of the CPU backend.
type CompiledFn struct {
	Name    string
	Code    []byte
	Rels    []Reloc
	IsEntry bool
}

// CompileModule lowers a MIR module to x86-64 machine code.
func CompileModule(m *mir.Module) ([]*CompiledFn, error) {
	var out []*CompiledFn
	for _, f := range m.Fns {
		cf, err := compileFn(f)
		if err != nil {
			return nil, err
		}
		out = append(out, cf)
	}
	return out, nil
}

// pendingJump records a jump instruction that needs patching after all code is emitted.
type pendingJump struct {
	offset    int  // offset in code where the 32-bit displacement starts
	labelID   int  // target label ID
	isCondJmp bool // true for conditional jump (6-byte), false for unconditional (5-byte)
}

type iselCtx struct {
	fn          *mir.Func
	alloc       *Allocation
	enc         *Encoder
	labels      map[int]int   // label ID -> code offset
	pendingJmps []pendingJump // jumps that need patching
	stackUsed   int32         // track dynamic stack usage for tensor allocations
	callIdx     int           // index of next OpCall in source order (for SaveAroundCall lookup)
}

func compileFn(f *mir.Func) (*CompiledFn, error) {
	alloc := Allocate(f)
	enc := &Encoder{}
	ctx := &iselCtx{
		fn:     f,
		alloc:  alloc,
		enc:    enc,
		labels: make(map[int]int),
	}

	// Calculate stack frame size needed for tensor allocations
	var tensorStackSize int64
	for _, in := range f.Body {
		if in.Op == mir.OpStackAlloc {
			tensorStackSize += in.Imm
		}
	}
	// Align to 16 bytes
	if tensorStackSize%16 != 0 {
		tensorStackSize += 16 - (tensorStackSize % 16)
	}

	// Prologue: set up stack frame if we have stack usage or calls
	hasCall := false
	hasTensors := tensorStackSize > 0
	for _, in := range f.Body {
		if in.Op == mir.OpCall {
			hasCall = true
			break
		}
	}
	needFrame := alloc.StackSize > 0 || hasCall || len(alloc.NeedsSave) > 0 || hasTensors || alloc.SaveAreaSize > 0

	// Lay out: [RSP+0 .. RSP+SaveAreaSize) = SIMD save slots,
	//         [RSP+SaveAreaSize .. RSP+SaveAreaSize+tensorStackSize) = OpStackAlloc tensors.
	// Spill-area uses negative offsets (existing convention) and is
	// not affected by SaveAreaSize.
	ctx.stackUsed = alloc.SaveAreaSize

	// For main, we always need a frame for tensor operations
	if needFrame && f.Name != "main" {
		// push rbp
		enc.Push(RBP)
		// mov rbp, rsp
		enc.MovRegToReg64(RBP, RSP)
		// Save callee-saved registers
		for _, r := range alloc.NeedsSave {
			enc.Push(r)
		}
		// Allocate stack space for locals + tensors + SIMD save area
		localStackSize := alloc.StackSize - int32(len(alloc.NeedsSave)*8) + int32(tensorStackSize) + alloc.SaveAreaSize
		if localStackSize > 0 {
			enc.SubImm32FromReg64(RSP, localStackSize)
		}
	} else if (hasTensors || alloc.SaveAreaSize > 0) && f.Name == "main" {
		// For main, we need stack space for tensors and/or SIMD save area
		enc.SubImm32FromReg64(RSP, int32(tensorStackSize)+alloc.SaveAreaSize)
	}

	// Move parameters from argument registers to allocated registers
	// We need to be careful about order to avoid clobbering
	for i := 0; i < len(f.Params); i++ {
		srcReg := SysVIntArgs[i]
		if dstReg, ok := alloc.GetReg(i); ok && dstReg != srcReg {
			enc.MovRegToReg64(dstReg, srcReg)
		}
	}

	// Canonicalise i8/u8 parameters: System V passes the value in the
	// low 8 bits of the argument register, but the upper bits are not
	// guaranteed. Sign- or zero-extend the low byte so subsequent
	// 64-bit operations see the canonical i8/u8 representation
	// (i8: sign-extended to 64; u8: zero-extended to 64).
	for i := 0; i < len(f.Params); i++ {
		ty := f.Params[i].Type
		if ty == nil {
			continue
		}
		dstReg, ok := alloc.GetReg(i)
		if !ok {
			continue
		}
		switch ty.K {
		case types.KI8:
			enc.MovsxReg8ToReg64(dstReg, dstReg)
		case types.KU8:
			enc.MovzxReg8ToReg64(dstReg, dstReg)
		}
	}

	// Generate code for each instruction
	for _, in := range f.Body {
		if err := ctx.emitInst(in); err != nil {
			return nil, err
		}
	}

	// Move result to correct return register
	if isFloatType(f.RetType) {
		// Float return: move to xmm0
		if xmm, ok := alloc.GetXMM(f.Result.ID); ok && xmm != XMM0 {
			enc.MovSSRegToReg(XMM0, xmm)
		}
	} else {
		// Integer return: move to RAX
		if dstReg, ok := alloc.GetReg(f.Result.ID); ok && dstReg != RAX {
			enc.MovRegToReg64(RAX, dstReg)
		}
	}

	if f.Name == "main" {
		// Restore stack if we allocated tensor space or SIMD save area
		if hasTensors || alloc.SaveAreaSize > 0 {
			enc.AddImm32ToReg64(RSP, int32(tensorStackSize)+alloc.SaveAreaSize)
		}
		// Exit syscall
		enc.MovRegToReg64(RDI, RAX)
		enc.MovImm32ToReg64(RAX, 60)
		enc.SyscallInsn()
	} else {
		// Epilogue
		if needFrame {
			// Deallocate local stack space + tensor space + SIMD save area
			localStackSize := alloc.StackSize - int32(len(alloc.NeedsSave)*8) + int32(tensorStackSize) + alloc.SaveAreaSize
			if localStackSize > 0 {
				enc.AddImm32ToReg64(RSP, localStackSize)
			}
			// Restore callee-saved registers in reverse order
			for i := len(alloc.NeedsSave) - 1; i >= 0; i-- {
				enc.Pop(alloc.NeedsSave[i])
			}
			// pop rbp
			enc.Pop(RBP)
		}
		enc.Ret()
	}

	// Patch jump offsets now that all labels are known
	if err := ctx.patchJumps(); err != nil {
		return nil, err
	}

	return &CompiledFn{
		Name:    f.Name,
		Code:    enc.Code,
		Rels:    enc.Rels,
		IsEntry: f.Name == "main",
	}, nil
}

// patchJumps fixes up jump instructions with correct offsets now that label positions are known.
func (ctx *iselCtx) patchJumps() error {
	for _, jmp := range ctx.pendingJmps {
		target, ok := ctx.labels[jmp.labelID]
		if !ok {
			return fmt.Errorf("isel: undefined label %d", jmp.labelID)
		}
		// Calculate relative offset from end of jump instruction
		// For JccRel32: instruction is 6 bytes (0F 8x + 4 byte offset), offset is at bytes 2-5
		// For JmpRel32: instruction is 5 bytes (E9 + 4 byte offset), offset is at byte 1-4
		var instrEnd int
		if jmp.isCondJmp {
			instrEnd = jmp.offset + 4 // 4 bytes of offset
		} else {
			instrEnd = jmp.offset + 4 // 4 bytes of offset
		}
		rel := int32(target - instrEnd)
		// Patch the 4-byte displacement in little-endian
		ctx.enc.Code[jmp.offset] = byte(rel)
		ctx.enc.Code[jmp.offset+1] = byte(rel >> 8)
		ctx.enc.Code[jmp.offset+2] = byte(rel >> 16)
		ctx.enc.Code[jmp.offset+3] = byte(rel >> 24)
	}
	return nil
}

// canonicaliseI8U8 ensures the canonical 64-bit representation of an
// i8 or u8 value is held in `reg`. i8 sign-extends the low byte;
// u8 zero-extends the low byte. For other types this is a no-op.
//
// This is invoked after integer arithmetic ops (+, -, *, /, %) whose
// result type is i8/u8, since those instructions leave junk in bits
// 8..63 of the result register.
func (ctx *iselCtx) canonicaliseI8U8(reg Reg, ty *types.Type) {
	if ty == nil {
		return
	}
	switch ty.K {
	case types.KI8:
		ctx.enc.MovsxReg8ToReg64(reg, reg)
	case types.KU8:
		ctx.enc.MovzxReg8ToReg64(reg, reg)
	}
}

func (ctx *iselCtx) emitInst(in mir.Inst) error {
	dst, dstInReg := ctx.alloc.GetReg(in.Result.ID)

	switch in.Op {
	case mir.OpConstInt:
		if !dstInReg {
			// Spilled: write the 32-bit constant directly into the value's
			// stack slot. Larger constants would need a two-store sequence;
			// reject for now since v0.10 only ever spills i32 consts.
			slot, ok := ctx.alloc.GetStack(in.Result.ID)
			if !ok {
				return fmt.Errorf("isel: OpConstInt result has no register and no stack slot")
			}
			if in.Imm < -0x80000000 || in.Imm > 0x7FFFFFFF {
				return fmt.Errorf("isel: spilled OpConstInt out of i32 range")
			}
			ctx.enc.MovImm32ToMem(RSP, slot, uint32(int32(in.Imm)))
			break
		}
		if in.Imm >= -0x80000000 && in.Imm <= 0x7FFFFFFF {
			ctx.enc.MovImm32ToReg64(dst, mustInt32(in.Imm))
		} else {
			ctx.enc.MovImm64ToReg64(dst, uint64(in.Imm))
		}

	case mir.OpConstFloat:
		isF64 := isF64Scalar(in.Result.Type)
		xmm, xmmInReg := ctx.alloc.GetXMM(in.Result.ID)
		if isF64 {
			bits := math.Float64bits(in.ImmF)
			if xmmInReg {
				ctx.enc.MovImm64ToMem(RSP, -8, bits)
				ctx.enc.MovSDLoadMem(xmm, RSP, -8)
				break
			}
			slot, ok := ctx.alloc.GetStack(in.Result.ID)
			if !ok {
				return fmt.Errorf("isel: OpConstFloat result has no register and no stack slot")
			}
			ctx.enc.MovImm64ToMem(RSP, slot, bits)
			break
		}
		bits := math.Float32bits(float32(in.ImmF))
		if xmmInReg {
			// Store float bits to a scratch stack location, then load via MOVSS.
			ctx.enc.MovImm32ToMem(RSP, -4, bits)
			ctx.enc.MovSSLoadMem(xmm, RSP, -4)
			break
		}
		// Spilled: write the bits directly into the value's stack slot.
		// Downstream consumers that need the value in an XMM will reload
		// it via the spill-aware paths below.
		slot, ok := ctx.alloc.GetStack(in.Result.ID)
		if !ok {
			return fmt.Errorf("isel: OpConstFloat result has no register and no stack slot")
		}
		ctx.enc.MovImm32ToMem(RSP, slot, bits)

	case mir.OpConstBool:
		if !dstInReg {
			// Spilled: write the constant directly into the value's stack
			// slot. Downstream consumers that need it in a register reload
			// from the slot.
			slot, ok := ctx.alloc.GetStack(in.Result.ID)
			if !ok {
				return fmt.Errorf("isel: OpConstBool result has no register and no stack slot")
			}
			var bits uint32
			if in.ImmB {
				bits = 1
			}
			ctx.enc.MovImm32ToMem(RSP, slot, bits)
			break
		}
		if in.ImmB {
			ctx.enc.MovImm32ToReg64(dst, 1)
		} else {
			ctx.enc.XorRegToReg64(dst, dst) // xor dst, dst -> 0
		}

	case mir.OpAdd:
		// Check if this is a float operation
		if isFloatType(in.Result.Type) {
			xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
			xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
			xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
			if !dstOk || !aOk || !bOk {
				return fmt.Errorf("isel: float add requires XMM registers")
			}
			if isF64Scalar(in.Result.Type) {
				if xmmDst != xmmA {
					ctx.enc.MovSDRegToReg(xmmDst, xmmA)
				}
				ctx.enc.AddSDRegToReg(xmmDst, xmmB)
				break
			}
			if xmmDst != xmmA {
				ctx.enc.MovSSRegToReg(xmmDst, xmmA)
			}
			ctx.enc.AddSSRegToReg(xmmDst, xmmB)
		} else {
			a := ctx.mustReg(in.Args[0].ID)
			b := ctx.mustReg(in.Args[1].ID)
			if !dstInReg {
				return fmt.Errorf("isel: spilled OpAdd result not yet supported (id=%d)", in.Result.ID)
			}
			if dst != a {
				ctx.enc.MovRegToReg64(dst, a)
			}
			ctx.enc.AddRegToReg64(dst, b)
			ctx.canonicaliseI8U8(dst, in.Result.Type)
		}

	case mir.OpSub:
		if isFloatType(in.Result.Type) {
			xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
			xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
			xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
			if !dstOk || !aOk || !bOk {
				return fmt.Errorf("isel: float sub requires XMM registers")
			}
			if isF64Scalar(in.Result.Type) {
				if xmmDst != xmmA {
					ctx.enc.MovSDRegToReg(xmmDst, xmmA)
				}
				ctx.enc.SubSDRegToReg(xmmDst, xmmB)
				break
			}
			if xmmDst != xmmA {
				ctx.enc.MovSSRegToReg(xmmDst, xmmA)
			}
			ctx.enc.SubSSRegToReg(xmmDst, xmmB)
		} else {
			a := ctx.mustReg(in.Args[0].ID)
			b := ctx.mustReg(in.Args[1].ID)
			if !dstInReg {
				return fmt.Errorf("isel: spilled OpSub result not yet supported (id=%d)", in.Result.ID)
			}
			if dst != a {
				ctx.enc.MovRegToReg64(dst, a)
			}
			ctx.enc.SubRegFromReg64(dst, b)
			ctx.canonicaliseI8U8(dst, in.Result.Type)
		}

	case mir.OpMul:
		if isFloatType(in.Result.Type) {
			xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
			xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
			xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
			if !dstOk || !aOk || !bOk {
				return fmt.Errorf("isel: float mul requires XMM registers")
			}
			if isF64Scalar(in.Result.Type) {
				if xmmDst != xmmA {
					ctx.enc.MovSDRegToReg(xmmDst, xmmA)
				}
				ctx.enc.MulSDRegToReg(xmmDst, xmmB)
				break
			}
			if xmmDst != xmmA {
				ctx.enc.MovSSRegToReg(xmmDst, xmmA)
			}
			ctx.enc.MulSSRegToReg(xmmDst, xmmB)
		} else {
			a := ctx.mustReg(in.Args[0].ID)
			b := ctx.mustReg(in.Args[1].ID)
			if !dstInReg {
				return fmt.Errorf("isel: spilled OpMul result not yet supported (id=%d)", in.Result.ID)
			}
			if dst != a {
				ctx.enc.MovRegToReg64(dst, a)
			}
			ctx.enc.IMulRegByReg64(dst, b)
			ctx.canonicaliseI8U8(dst, in.Result.Type)
		}

	case mir.OpDiv, mir.OpMod:
		if isFloatType(in.Result.Type) {
			if in.Op == mir.OpMod {
				return fmt.Errorf("isel: float modulo not supported")
			}
			xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
			xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
			xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
			if !dstOk || !aOk || !bOk {
				return fmt.Errorf("isel: float div requires XMM registers")
			}
			if isF64Scalar(in.Result.Type) {
				if xmmDst != xmmA {
					ctx.enc.MovSDRegToReg(xmmDst, xmmA)
				}
				ctx.enc.DivSDRegToReg(xmmDst, xmmB)
				break
			}
			if xmmDst != xmmA {
				ctx.enc.MovSSRegToReg(xmmDst, xmmA)
			}
			ctx.enc.DivSSRegToReg(xmmDst, xmmB)
		} else {
			a := ctx.mustReg(in.Args[0].ID)
			b := ctx.mustReg(in.Args[1].ID)
			if !dstInReg {
				return fmt.Errorf("isel: spilled OpDiv/OpMod result not yet supported (id=%d)", in.Result.ID)
			}
			// Division uses RAX for dividend and quotient, RDX for remainder.
			// For signed types we sign-extend via CQO and use IDIV.
			// For unsigned types (u32, u64) we zero RDX and use DIV so the
			// quotient is interpreted as an unsigned value.
			if a != RAX {
				ctx.enc.MovRegToReg64(RAX, a)
			}
			if in.Result.Type.IsUnsigned() {
				ctx.enc.XorRegToReg64(RDX, RDX) // zero RDX for unsigned divide
				ctx.enc.DivReg64(b)
			} else {
				ctx.enc.CQO() // sign-extend RAX into RDX:RAX
				ctx.enc.IDivReg64(b)
			}
			// Quotient in RAX, remainder in RDX
			if in.Op == mir.OpDiv {
				if dst != RAX {
					ctx.enc.MovRegToReg64(dst, RAX)
				}
			} else {
				if dst != RDX {
					ctx.enc.MovRegToReg64(dst, RDX)
				}
			}
			ctx.canonicaliseI8U8(dst, in.Result.Type)
		}

	case mir.OpNeg:
		if isFloatType(in.Result.Type) {
			// Float negation: xor with sign bit mask, or subtract from 0
			// Simpler: load 0, subtract the value
			xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
			xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
			if !dstOk || !aOk {
				return fmt.Errorf("isel: float neg requires XMM registers")
			}
			// Load 0.0 into dst, then subtract a
			ctx.enc.XorPSRegToReg(xmmDst, xmmDst) // dst = 0
			ctx.enc.SubSSRegToReg(xmmDst, xmmA)   // dst = 0 - a
		} else {
			a := ctx.mustReg(in.Args[0].ID)
			if !dstInReg {
				return fmt.Errorf("isel: spilled OpNeg result not yet supported (id=%d)", in.Result.ID)
			}
			if dst != a {
				ctx.enc.MovRegToReg64(dst, a)
			}
			ctx.enc.NegReg64(dst)
		}

	case mir.OpNot:
		a := ctx.mustReg(in.Args[0].ID)
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpNot result not yet supported (id=%d)", in.Result.ID)
		}
		// Logical not: x -> !x (1 - x for bool values 0/1)
		// xor dst, dst; test a, a; sete dst
		ctx.enc.XorRegToReg64(dst, dst)
		ctx.enc.TestRegToReg64(a, a)
		ctx.enc.SetCC(CCE, dst) // sete (set if zero)
		// Now dst has 0 or 1 in low byte; zero-extend
		ctx.enc.MovzxReg8ToReg64(dst, dst)

	case mir.OpEq, mir.OpNe, mir.OpLt, mir.OpLe, mir.OpGt, mir.OpGe:
		a := ctx.mustReg(in.Args[0].ID)
		b := ctx.mustReg(in.Args[1].ID)
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpCmp result not yet supported (op=%v id=%d)", in.Op, in.Result.ID)
		}
		// Equality is signedness-agnostic; orderings dispatch on the
		// operand type. We pick on Args[0].Type because comparisons
		// produce bool, so in.Result.Type wouldn't carry the operand
		// signedness.
		unsigned := len(in.Args) > 0 && in.Args[0].Type.IsUnsigned()
		var cc CondCode
		switch in.Op {
		case mir.OpEq:
			cc = CCE
		case mir.OpNe:
			cc = CCNE
		case mir.OpLt:
			if unsigned {
				cc = CCB
			} else {
				cc = CCL
			}
		case mir.OpLe:
			if unsigned {
				cc = CCBE
			} else {
				cc = CCLE
			}
		case mir.OpGt:
			if unsigned {
				cc = CCA
			} else {
				cc = CCG
			}
		case mir.OpGe:
			if unsigned {
				cc = CCAE
			} else {
				cc = CCGE
			}
		}
		// XOR must come BEFORE CMP to avoid clobbering flags
		ctx.enc.XorRegToReg64(dst, dst)
		ctx.enc.CmpRegToReg64(a, b)
		ctx.enc.SetCC(cc, dst)
		ctx.enc.MovzxReg8ToReg64(dst, dst)

	case mir.OpAnd:
		a := ctx.mustReg(in.Args[0].ID)
		b := ctx.mustReg(in.Args[1].ID)
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpAnd result not yet supported (id=%d)", in.Result.ID)
		}
		if dst != a {
			ctx.enc.MovRegToReg64(dst, a)
		}
		ctx.enc.AndRegToReg64(dst, b)

	case mir.OpOr:
		a := ctx.mustReg(in.Args[0].ID)
		b := ctx.mustReg(in.Args[1].ID)
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpOr result not yet supported (id=%d)", in.Result.ID)
		}
		if dst != a {
			ctx.enc.MovRegToReg64(dst, a)
		}
		ctx.enc.OrRegToReg64(dst, b)

	case mir.OpSelect:
		// select(cond, t, f) -> cond ? t : f
		cond := ctx.mustReg(in.Args[0].ID)
		tVal := ctx.mustReg(in.Args[1].ID)
		fVal := ctx.mustReg(in.Args[2].ID)
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpSelect result not yet supported (id=%d)", in.Result.ID)
		}
		// Move false value to dst, then conditionally move true value
		if dst != fVal {
			ctx.enc.MovRegToReg64(dst, fVal)
		}
		ctx.enc.TestRegToReg64(cond, cond)
		ctx.enc.CMov(CCNE, dst, tVal) // cmovne dst, tVal (if cond != 0)

	case mir.OpCopy:
		if isFloatType(in.Result.Type) {
			xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
			xmmSrc, srcOk := ctx.alloc.GetXMM(in.Args[0].ID)
			if !dstOk || !srcOk {
				return fmt.Errorf("isel: float copy requires XMM registers")
			}
			if xmmDst != xmmSrc {
				ctx.enc.MovSSRegToReg(xmmDst, xmmSrc)
			}
		} else {
			// GPR copy: handle spilled source and/or destination.
			srcReg, srcInReg := ctx.alloc.GetReg(in.Args[0].ID)
			if srcInReg && dstInReg {
				if dst != srcReg {
					ctx.enc.MovRegToReg64(dst, srcReg)
				}
			} else if srcInReg && !dstInReg {
				slot, ok := ctx.alloc.GetStack(in.Result.ID)
				if !ok {
					return fmt.Errorf("isel: spilled OpCopy result has no stack slot")
				}
				ctx.enc.MovStoreMem64(RSP, slot, srcReg)
			} else if !srcInReg && dstInReg {
				slot, ok := ctx.alloc.GetStack(in.Args[0].ID)
				if !ok {
					return fmt.Errorf("isel: spilled OpCopy src has no stack slot")
				}
				ctx.enc.MovLoadMem64(dst, RSP, slot)
			} else {
				// Both spilled: use RAX as scratch.
				srcSlot, ok := ctx.alloc.GetStack(in.Args[0].ID)
				if !ok {
					return fmt.Errorf("isel: spilled OpCopy src has no stack slot")
				}
				dstSlot, ok := ctx.alloc.GetStack(in.Result.ID)
				if !ok {
					return fmt.Errorf("isel: spilled OpCopy result has no stack slot")
				}
				ctx.enc.MovLoadMem64(RAX, RSP, srcSlot)
				ctx.enc.MovStoreMem64(RSP, dstSlot, RAX)
			}
		}

	case mir.OpCall:
		// Save SIMD values that are live across this call. All XMM/YMM
		// registers are caller-saved on System V AMD64; without this
		// pass any f32 / Vec4F32 / Vec8F32 value used after the call
		// would be clobbered by the callee. Save slots were reserved
		// by regalloc; emit the stores BEFORE arg setup so the arg
		// registers (xmm0..7) can be written without losing values.
		var saves []CallSaveEntry
		if ctx.callIdx < len(ctx.alloc.CallSaves) {
			saves = ctx.alloc.CallSaves[ctx.callIdx]
		}
		for _, e := range saves {
			switch e.Kind {
			case SaveXMM:
				ctx.enc.MovUPSStore128(RSP, e.Slot, e.XMM)
			case SaveYMM:
				ctx.enc.VMovUPSStore256(RSP, e.Slot, e.YMM)
			}
		}

		// Set up arguments in registers per SysV: int/pointer args fill
		// rdi, rsi, rdx, rcx, r8, r9 in order; float args fill xmm0..7
		// in their own counter. For each arg we dispatch on its type.
		intIdx, floatIdx := 0, 0
		for _, arg := range in.Args {
			if isFloatType(arg.Type) {
				if floatIdx >= len(SysVFloatArgs) {
					return fmt.Errorf("isel: too many float call arguments (stack args not yet supported)")
				}
				srcXMM := ctx.mustXMM(arg.ID)
				dstXMM := SysVFloatArgs[floatIdx]
				if srcXMM != dstXMM {
					ctx.enc.MovSSRegToReg(dstXMM, srcXMM)
				}
				floatIdx++
				continue
			}
			if intIdx >= len(SysVIntArgs) {
				return fmt.Errorf("isel: too many int call arguments (stack args not yet supported)")
			}
			srcReg := ctx.mustReg(arg.ID)
			dstReg := SysVIntArgs[intIdx]
			if srcReg != dstReg {
				ctx.enc.MovRegToReg64(dstReg, srcReg)
			}
			intIdx++
		}
		ctx.enc.CallSym(in.FnName)
		// Move result from return register to allocated register
		if isFloatType(in.Result.Type) {
			// Float return: result in xmm0
			xmm, xmmOk := ctx.alloc.GetXMM(in.Result.ID)
			if xmmOk && xmm != XMM0 {
				ctx.enc.MovSSRegToReg(xmm, XMM0)
			}
		} else {
			// Integer return: result in RAX
			if dstInReg && dst != RAX {
				ctx.enc.MovRegToReg64(dst, RAX)
			}
		}

		// Restore SIMD values clobbered by the call. If the result of
		// the call itself is among the saved registers (e.g. a value
		// allocated to XMM_k where the just-returned XMM_k holds the
		// callee's result mov'd from XMM0 above), reloading from the
		// save slot would overwrite the return value with the
		// pre-call content. Skip any save entry whose XMM matches
		// the result's allocated register.
		var resultXMM XMMReg
		resultXMMValid := false
		if isFloatType(in.Result.Type) {
			if xmm, ok := ctx.alloc.GetXMM(in.Result.ID); ok {
				resultXMM = xmm
				resultXMMValid = true
			}
		}
		for _, e := range saves {
			switch e.Kind {
			case SaveXMM:
				if resultXMMValid && e.XMM == resultXMM {
					continue
				}
				ctx.enc.MovUPSLoad128(e.XMM, RSP, e.Slot)
			case SaveYMM:
				ctx.enc.VMovUPSLoad256(e.YMM, RSP, e.Slot)
			}
		}
		ctx.callIdx++

	case mir.OpStackAlloc:
		// Allocate space on stack and return pointer
		// The actual allocation is done in prologue; here we compute the address
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpStackAlloc not yet supported")
		}
		// Use LEA to get address relative to RSP
		// Stack grows downward; allocations are at positive offsets from RSP
		ctx.enc.LEA(dst, RSP, ctx.stackUsed)
		ctx.stackUsed += int32(in.Imm)

	case mir.OpRodataPtr:
		// Load the address of a .rodata symbol into a GPR via a single
		// RIP-relative LEA. The relocation is recorded by the encoder
		// and resolved at link time.
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpRodataPtr not yet supported")
		}
		ctx.enc.LeaRipSym(dst, in.FnName)

	case mir.OpLoad32:
		// Load f32 from [base + offset]
		xmm, xmmInReg := ctx.alloc.GetXMM(in.Result.ID)
		if !xmmInReg {
			return fmt.Errorf("isel: OpLoad32 result must be in XMM register")
		}
		base := ctx.mustReg(in.Args[0].ID)
		ctx.enc.MovSSLoadMem(xmm, base, int32(in.Imm))

	case mir.OpStore32:
		// Store f32 to [base + offset]. If the value was spilled (regalloc
		// ran out of XMM slots and assigned a stack slot), reload it
		// through XMM0 as a scratch first.
		base := ctx.mustReg(in.Args[0].ID)
		xmm, xmmInReg := ctx.alloc.GetXMM(in.Args[1].ID)
		if !xmmInReg {
			slot, ok := ctx.alloc.GetStack(in.Args[1].ID)
			if !ok {
				return fmt.Errorf("isel: OpStore32 value has neither XMM nor stack slot")
			}
			ctx.enc.MovSSLoadMem(XMM0, RSP, slot)
			xmm = XMM0
		}
		ctx.enc.MovSSStoreMem(base, int32(in.Imm), xmm)

	case mir.OpLoadI32:
		// Load i32 from [base + offset] into a GPR (zero-extended to 64).
		if !dstInReg {
			return fmt.Errorf("isel: OpLoadI32 result must be in GPR")
		}
		base := ctx.mustReg(in.Args[0].ID)
		ctx.enc.MovLoadMem32(dst, base, int32(in.Imm))

	case mir.OpStoreI32:
		// Store i32 to [base + offset] from a GPR.
		base := ctx.mustReg(in.Args[0].ID)
		src := ctx.mustReg(in.Args[1].ID)
		ctx.enc.MovStoreMem32(base, int32(in.Imm), src)

	case mir.OpLoad64:
		// Load i64 from [base + offset]
		if !dstInReg {
			return fmt.Errorf("isel: OpLoad64 result must be in GPR")
		}
		base := ctx.mustReg(in.Args[0].ID)
		ctx.enc.MovLoadMem64(dst, base, int32(in.Imm))

	case mir.OpStore64:
		// Store i64 to [base + offset]
		base := ctx.mustReg(in.Args[0].ID)
		src := ctx.mustReg(in.Args[1].ID)
		ctx.enc.MovStoreMem64(base, int32(in.Imm), src)

	case mir.OpLEA:
		// Load effective address
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpLEA not yet supported")
		}
		base := ctx.mustReg(in.Args[0].ID)
		ctx.enc.LEA(dst, base, int32(in.Imm))

	case mir.OpAddF32:
		xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
		xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
		xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
		if !dstOk || !aOk || !bOk {
			return fmt.Errorf("isel: OpAddF32 requires XMM registers")
		}
		if xmmDst != xmmA {
			ctx.enc.MovSSRegToReg(xmmDst, xmmA)
		}
		ctx.enc.AddSSRegToReg(xmmDst, xmmB)

	case mir.OpSubF32:
		xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
		xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
		xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
		if !dstOk || !aOk || !bOk {
			return fmt.Errorf("isel: OpSubF32 requires XMM registers")
		}
		if xmmDst != xmmA {
			ctx.enc.MovSSRegToReg(xmmDst, xmmA)
		}
		ctx.enc.SubSSRegToReg(xmmDst, xmmB)

	case mir.OpMulF32:
		xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
		xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
		xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
		if !dstOk || !aOk || !bOk {
			return fmt.Errorf("isel: OpMulF32 requires XMM registers")
		}
		if xmmDst != xmmA {
			ctx.enc.MovSSRegToReg(xmmDst, xmmA)
		}
		ctx.enc.MulSSRegToReg(xmmDst, xmmB)

	case mir.OpDivF32:
		xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
		xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
		xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
		if !dstOk || !aOk || !bOk {
			return fmt.Errorf("isel: OpDivF32 requires XMM registers")
		}
		if xmmDst != xmmA {
			ctx.enc.MovSSRegToReg(xmmDst, xmmA)
		}
		ctx.enc.DivSSRegToReg(xmmDst, xmmB)

	case mir.OpMaxF32:
		xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
		xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
		xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
		if !dstOk || !aOk || !bOk {
			return fmt.Errorf("isel: OpMaxF32 requires XMM registers")
		}
		if xmmDst != xmmA {
			ctx.enc.MovSSRegToReg(xmmDst, xmmA)
		}
		ctx.enc.MaxSSRegToReg(xmmDst, xmmB)

	case mir.OpMinF32:
		xmmDst, dstOk := ctx.alloc.GetXMM(in.Result.ID)
		xmmA, aOk := ctx.alloc.GetXMM(in.Args[0].ID)
		xmmB, bOk := ctx.alloc.GetXMM(in.Args[1].ID)
		if !dstOk || !aOk || !bOk {
			return fmt.Errorf("isel: OpMinF32 requires XMM registers")
		}
		if xmmDst != xmmA {
			ctx.enc.MovSSRegToReg(xmmDst, xmmA)
		}
		ctx.enc.MinSSRegToReg(xmmDst, xmmB)

	case mir.OpLabel:
		// Record label position
		ctx.labels[in.LabelID] = len(ctx.enc.Code)

	case mir.OpCmpLt:
		// Compare for less than (integer). Dispatch on operand
		// signedness: u32/u64 use CCB (CF=1), signed types use CCL.
		a := ctx.mustReg(in.Args[0].ID)
		b := ctx.mustReg(in.Args[1].ID)
		if !dstInReg {
			return fmt.Errorf("isel: spilled OpCmpLt not yet supported")
		}
		cc := CCL
		if len(in.Args) > 0 && in.Args[0].Type.IsUnsigned() {
			cc = CCB
		}
		ctx.enc.XorRegToReg64(dst, dst)
		ctx.enc.CmpRegToReg64(a, b)
		ctx.enc.SetCC(cc, dst)
		ctx.enc.MovzxReg8ToReg64(dst, dst)

	case mir.OpJumpIf:
		// Conditional jump based on comparison result
		// Args[0] is a boolean from a comparison
		// We need to test the boolean and jump if true (non-zero)
		cond := ctx.mustReg(in.Args[0].ID)
		// test reg, reg (sets ZF if zero)
		ctx.enc.TestRegToReg64(cond, cond)
		// Emit JNE (jump if not zero, i.e., condition is true) with placeholder offset
		jmpOffset := len(ctx.enc.Code)
		ctx.enc.JccRel32(CCNE, 0) // placeholder offset
		ctx.pendingJmps = append(ctx.pendingJmps, pendingJump{
			offset:    jmpOffset + 2, // skip 0F 85 opcode bytes
			labelID:   in.LabelID,
			isCondJmp: true,
		})

	case mir.OpJump:
		// Unconditional jump
		jmpOffset := len(ctx.enc.Code)
		ctx.enc.JmpRel32(0) // placeholder offset
		ctx.pendingJmps = append(ctx.pendingJmps, pendingJump{
			offset:    jmpOffset + 1, // skip E9 opcode byte
			labelID:   in.LabelID,
			isCondJmp: false,
		})

	case mir.OpCastF32ToI32:
		// Convert f32 (XMM) to i32 (GPR) with truncation
		xmm, xmmOk := ctx.alloc.GetXMM(in.Args[0].ID)
		if !xmmOk {
			return fmt.Errorf("isel: OpCastF32ToI32 source must be in XMM register")
		}
		if !dstInReg {
			return fmt.Errorf("isel: OpCastF32ToI32 result must be in GPR")
		}
		ctx.enc.CVTTSS2SI(dst, xmm)

	case mir.OpCastI32ToF32:
		// Convert i32 (GPR) to f32 (XMM)
		src := ctx.mustReg(in.Args[0].ID)
		xmm, xmmOk := ctx.alloc.GetXMM(in.Result.ID)
		if !xmmOk {
			return fmt.Errorf("isel: OpCastI32ToF32 result must be in XMM register")
		}
		ctx.enc.CVTSI2SS(xmm, src)

	case mir.OpCastF64ToI32:
		// Convert f64 (XMM) to i32 (GPR) with truncation toward zero.
		xmm, xmmOk := ctx.alloc.GetXMM(in.Args[0].ID)
		if !xmmOk {
			return fmt.Errorf("isel: OpCastF64ToI32 source must be in XMM register")
		}
		if !dstInReg {
			return fmt.Errorf("isel: OpCastF64ToI32 result must be in GPR")
		}
		ctx.enc.CVTTSD2SI(dst, xmm)

	case mir.OpCastI32ToF64:
		// Convert i32 (GPR) to f64 (XMM).
		src := ctx.mustReg(in.Args[0].ID)
		xmm, xmmOk := ctx.alloc.GetXMM(in.Result.ID)
		if !xmmOk {
			return fmt.Errorf("isel: OpCastI32ToF64 result must be in XMM register")
		}
		ctx.enc.CVTSI2SD(xmm, src)

	case mir.OpCastI8ToI32, mir.OpCastI8ToI64,
		mir.OpCastI64ToI8, mir.OpCastI32ToI8:
		// Sign-extend the low 8 bits of the source register into dst.
		// Truncating casts (i32→i8, i64→i8) and widening casts
		// (i8→i32, i8→i64) all collapse to the same MOVSX r64, r8
		// because the canonical i8 invariant means the source's low
		// byte already holds the sign-extended value.
		src := ctx.mustReg(in.Args[0].ID)
		if !dstInReg {
			return fmt.Errorf("isel: signed i8 cast result not in register (id=%d)", in.Result.ID)
		}
		ctx.enc.MovsxReg8ToReg64(dst, src)

	case mir.OpCastU8ToI32, mir.OpCastU8ToI64,
		mir.OpCastI64ToU8, mir.OpCastI32ToU8:
		// Zero-extend the low 8 bits of the source register into dst.
		src := ctx.mustReg(in.Args[0].ID)
		if !dstInReg {
			return fmt.Errorf("isel: unsigned u8 cast result not in register (id=%d)", in.Result.ID)
		}
		ctx.enc.MovzxReg8ToReg64(dst, src)

	// AVX2 packed f32 operations
	case mir.OpLoad256:
		ymm := ctx.mustYMM(in.Result.ID)
		base := ctx.mustReg(in.Args[0].ID)
		ctx.enc.VMovUPSLoad256(ymm, base, int32(in.Imm))

	case mir.OpStore256:
		base := ctx.mustReg(in.Args[0].ID)
		ymm := ctx.mustYMM(in.Args[1].ID)
		ctx.enc.VMovUPSStore256(base, int32(in.Imm), ymm)

	case mir.OpAddPS256:
		dst := ctx.mustYMM(in.Result.ID)
		src1 := ctx.mustYMM(in.Args[0].ID)
		src2 := ctx.mustYMM(in.Args[1].ID)
		ctx.enc.VAddPS256(dst, src1, src2)

	case mir.OpSubPS256:
		dst := ctx.mustYMM(in.Result.ID)
		src1 := ctx.mustYMM(in.Args[0].ID)
		src2 := ctx.mustYMM(in.Args[1].ID)
		ctx.enc.VSubPS256(dst, src1, src2)

	case mir.OpMulPS256:
		dst := ctx.mustYMM(in.Result.ID)
		src1 := ctx.mustYMM(in.Args[0].ID)
		src2 := ctx.mustYMM(in.Args[1].ID)
		ctx.enc.VMulPS256(dst, src1, src2)

	case mir.OpDivPS256:
		dst := ctx.mustYMM(in.Result.ID)
		src1 := ctx.mustYMM(in.Args[0].ID)
		src2 := ctx.mustYMM(in.Args[1].ID)
		ctx.enc.VDivPS256(dst, src1, src2)

	// SSE packed f32 operations (4 elements)
	case mir.OpLoad128:
		xmm := ctx.mustXMM(in.Result.ID)
		base := ctx.mustReg(in.Args[0].ID)
		ctx.enc.MovUPSLoad128(xmm, base, int32(in.Imm))

	case mir.OpStore128:
		base := ctx.mustReg(in.Args[0].ID)
		xmm := ctx.mustXMM(in.Args[1].ID)
		ctx.enc.MovUPSStore128(base, int32(in.Imm), xmm)

	case mir.OpAddPS128:
		dst := ctx.mustXMM(in.Result.ID)
		src1 := ctx.mustXMM(in.Args[0].ID)
		src2 := ctx.mustXMM(in.Args[1].ID)
		if dst != src1 {
			ctx.enc.MovUPSReg128(dst, src1)
		}
		ctx.enc.AddPSReg128(dst, src2)

	case mir.OpSubPS128:
		dst := ctx.mustXMM(in.Result.ID)
		src1 := ctx.mustXMM(in.Args[0].ID)
		src2 := ctx.mustXMM(in.Args[1].ID)
		if dst != src1 {
			ctx.enc.MovUPSReg128(dst, src1)
		}
		ctx.enc.SubPSReg128(dst, src2)

	case mir.OpMulPS128:
		dst := ctx.mustXMM(in.Result.ID)
		src1 := ctx.mustXMM(in.Args[0].ID)
		src2 := ctx.mustXMM(in.Args[1].ID)
		if dst != src1 {
			ctx.enc.MovUPSReg128(dst, src1)
		}
		ctx.enc.MulPSReg128(dst, src2)

	case mir.OpDivPS128:
		dst := ctx.mustXMM(in.Result.ID)
		src1 := ctx.mustXMM(in.Args[0].ID)
		src2 := ctx.mustXMM(in.Args[1].ID)
		if dst != src1 {
			ctx.enc.MovUPSReg128(dst, src1)
		}
		ctx.enc.DivPSReg128(dst, src2)

	// Horizontal SIMD reduction primitives.
	// HAddPS128 has a 2-source form at the SSE level (haddps xmm, xmm), so
	// we must move src1 into dst first if they differ.
	case mir.OpHAddPS128:
		dst := ctx.mustXMM(in.Result.ID)
		src1 := ctx.mustXMM(in.Args[0].ID)
		src2 := ctx.mustXMM(in.Args[1].ID)
		if dst != src1 {
			ctx.enc.MovUPSReg128(dst, src1)
		}
		ctx.enc.HAddPSReg128(dst, src2)

	case mir.OpHAddPS256:
		dst := ctx.mustYMM(in.Result.ID)
		src1 := ctx.mustYMM(in.Args[0].ID)
		src2 := ctx.mustYMM(in.Args[1].ID)
		ctx.enc.VHAddPS256(dst, src1, src2)

	case mir.OpExtractF128Lo:
		dst := ctx.mustXMM(in.Result.ID)
		src := ctx.mustYMM(in.Args[0].ID)
		ctx.enc.VExtractF128(dst, src, 0)

	case mir.OpExtractF128Hi:
		dst := ctx.mustXMM(in.Result.ID)
		src := ctx.mustYMM(in.Args[0].ID)
		ctx.enc.VExtractF128(dst, src, 1)

	default:
		return fmt.Errorf("isel: unsupported op %d", in.Op)
	}

	return nil
}

func (ctx *iselCtx) mustReg(id int) Reg {
	r, ok := ctx.alloc.GetReg(id)
	if !ok {
		panic(fmt.Sprintf("isel: value %d not in register", id))
	}
	return r
}

func (ctx *iselCtx) mustXMM(id int) XMMReg {
	r, ok := ctx.alloc.GetXMM(id)
	if !ok {
		panic(fmt.Sprintf("isel: value %d not in XMM register", id))
	}
	return r
}

func (ctx *iselCtx) mustYMM(id int) YMMReg {
	r, ok := ctx.alloc.GetYMM(id)
	if !ok {
		panic(fmt.Sprintf("isel: value %d not in YMM register", id))
	}
	return r
}

// isFloatType checks if a type should use XMM registers.
// Tensor types use GPR registers (they're pointers), not XMM.
func isFloatType(t *types.Type) bool {
	if t == nil {
		return false
	}
	// Tensors are passed as pointers, not in XMM registers
	if t.IsTensor() {
		return false
	}
	return t.K == types.KF32 || t.K == types.KF64
}

// isF64Scalar reports whether t is the scalar f64 type. Drives SS-vs-SD
// instruction selection for the float arithmetic and constant-load
// paths.
func isF64Scalar(t *types.Type) bool {
	return t != nil && !t.IsTensor() && t.K == types.KF64
}
