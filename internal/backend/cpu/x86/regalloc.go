package x86

import (
	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/types"
)

// Allocation maps MIR SSA value IDs to physical registers or stack slots.
type Allocation struct {
	Map        map[int]Reg
	XMMMap     map[int]XMMReg // for float values
	YMMMap     map[int]YMMReg // for vectorized values (8 x f32)
	StackSlots map[int]int32
	StackSize  int32
	// NeedsSave lists callee-saved registers we use and must save/restore.
	NeedsSave []Reg

	// SIMD save/restore around calls.
	//
	// All XMM/YMM registers are caller-saved on System V AMD64. Any
	// SIMD value (scalar f32, Vec4F32 in XMM, Vec8F32 in YMM) that
	// remains live across an OpCall would be clobbered by the callee.
	// liveAcrossCall already marks them, but the SIMD branch of the
	// linear scan can't move them to a "callee-saved" SIMD register
	// (none exist). Instead we reserve a stack slot per such value
	// and emit save/restore around each call site that crosses it.
	//
	// The save area lives at positive offsets [0, SaveAreaSize) from
	// RSP after the prologue's `sub rsp, ...`. Slots are 16-byte
	// aligned (XMM) or 32-byte aligned (YMM). isel sets
	// ctx.stackUsed = SaveAreaSize on entry so OpStackAlloc tensor
	// allocations sit above the save area.
	//
	// CallSaves[i] is the entry list to save before / restore after
	// the i-th OpCall in source order.
	SaveAreaSize int32
	CallSaves    [][]CallSaveEntry
}

// CallSaveEntry is one SIMD value to save/restore around a call site.
type CallSaveEntry struct {
	Slot int32 // positive RSP-relative offset
	Kind CallSaveKind
	XMM  XMMReg
	YMM  YMMReg
}

// CallSaveKind selects the right save/restore encoding.
type CallSaveKind uint8

const (
	SaveXMM CallSaveKind = iota // 16 bytes via movups
	SaveYMM                     // 32 bytes via vmovups
)

// Callee-saved registers that we use for allocation. These survive calls.
var calleeSavedPool = []Reg{RBX, R12, R13, R14, R15}

// Allocate assigns registers to all SSA values defined in fn.Body.
// Uses liveness analysis to reuse registers when values die.
func Allocate(fn *mir.Func) *Allocation {
	a := &Allocation{
		Map:        map[int]Reg{},
		XMMMap:     map[int]XMMReg{},
		YMMMap:     map[int]YMMReg{},
		StackSlots: map[int]int32{},
	}

	// Compute last use for each value
	lastUse := map[int]int{} // value ID -> last instruction index that uses it
	for i, in := range fn.Body {
		for _, arg := range in.Args {
			lastUse[arg.ID] = i
		}
	}
	// Function result is used "after" all instructions
	lastUse[fn.Result.ID] = len(fn.Body)

	// Defined-but-never-used values still claim a register at their
	// definition site. Without an entry in lastUse, freeDeadRegs (which
	// reclaims registers where lastUse == i-1) never fires for them and
	// they leak across the rest of the function. Seed lastUse with the
	// definition index so the register is reclaimed on the next inst.
	for i, in := range fn.Body {
		if in.Result.ID == 0 {
			continue
		}
		if _, seen := lastUse[in.Result.ID]; !seen {
			lastUse[in.Result.ID] = i
		}
	}

	// Loop-aware lastUse extension. Linear-scan walks the MIR in source
	// order and records the maximum index where each value appears as
	// an argument. That ignores back-edges: a value defined before a
	// loop and used inside the body has its interval end at the in-body
	// use, after which the regalloc may reuse its register for a
	// later in-body value. The next runtime iteration jumps back to
	// the loop top and finds the register clobbered.
	//
	// Detect back-edges (Jump/JumpIf whose target label sits at a lower
	// instruction index) and, for every value defined strictly before
	// the loop top that is used anywhere inside the loop body, extend
	// its lastUse to the back-edge index. This pins the register
	// across the entire loop region.
	labelPos := map[int]int{}
	for i, in := range fn.Body {
		if in.Op == mir.OpLabel {
			labelPos[in.LabelID] = i
		}
	}
	definedAt := map[int]int{}
	for i := 0; i < len(fn.Params); i++ {
		definedAt[i] = -1
	}
	for i, in := range fn.Body {
		definedAt[in.Result.ID] = i
	}
	for i, in := range fn.Body {
		if in.Op != mir.OpJump && in.Op != mir.OpJumpIf {
			continue
		}
		targetIdx, ok := labelPos[in.LabelID]
		if !ok || targetIdx >= i {
			continue
		}
		// Loop region is [targetIdx .. i].
		for j := targetIdx; j <= i; j++ {
			for _, arg := range fn.Body[j].Args {
				def, ok := definedAt[arg.ID]
				if !ok {
					continue
				}
				if def < targetIdx {
					if cur, seen := lastUse[arg.ID]; !seen || cur < i {
						lastUse[arg.ID] = i
					}
				}
			}
		}
	}

	// Check if function has any calls and find call positions
	callPositions := []int{}
	for i, in := range fn.Body {
		if in.Op == mir.OpCall {
			callPositions = append(callPositions, i)
		}
	}
	hasCall := len(callPositions) > 0

	// Find values that are live across any call
	liveAcrossCall := map[int]bool{}
	if hasCall {
		defined := map[int]int{} // value ID -> instruction index where defined
		for i, in := range fn.Body {
			// Many ops (Store*, Jump*, Label) leave Result.ID=0 as a
			// "no SSA result" sentinel. Don't overwrite a real
			// definition (especially the very first instruction whose
			// genuine Result.ID may be 0) with a later sink op.
			id := in.Result.ID
			if _, seen := defined[id]; seen {
				continue
			}
			defined[id] = i
		}
		for i := 0; i < len(fn.Params); i++ {
			defined[i] = -1
		}

		for _, callIdx := range callPositions {
			for id, defIdx := range defined {
				if defIdx < callIdx && lastUse[id] > callIdx {
					liveAcrossCall[id] = true
				}
			}
		}
	}

	// Register pools with availability tracking
	callerSavedPool := []Reg{RCX, R8, R9, R10, R11}
	callerSavedFree := make([]bool, len(callerSavedPool))
	for i := range callerSavedFree {
		callerSavedFree[i] = true
	}

	calleeSavedFree := make([]bool, len(calleeSavedPool))
	calleeSavedEverUsed := make([]bool, len(calleeSavedPool))
	for i := range calleeSavedFree {
		calleeSavedFree[i] = true
		calleeSavedEverUsed[i] = false
	}

	// XMM/YMM share the same physical SIMD register file: xmm_k aliases
	// the low 128 bits of ymm_k. Track availability with a single shared
	// array indexed by k. xmmPool[i] and ymmPool[i] refer to the same
	// physical register; allocating one must mark the other busy.
	// XMM0/YMM0 reserved for return value.
	xmmPool := []XMMReg{XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15}
	ymmPool := []YMMReg{YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7, YMM8, YMM9, YMM10, YMM11, YMM12, YMM13, YMM14, YMM15}
	simdFree := make([]bool, len(xmmPool))
	for i := range simdFree {
		simdFree[i] = true
	}
	xmmFree := simdFree
	ymmFree := simdFree

	// Track which register is assigned to which value
	regToValue := map[Reg]int{}
	xmmToValue := map[XMMReg]int{}
	ymmToValue := map[YMMReg]int{}

	// Helper to free registers for values that died at instruction i
	freeDeadRegs := func(i int) {
		for id, lu := range lastUse {
			if lu == i-1 {
				if reg, ok := a.Map[id]; ok {
					for j, r := range callerSavedPool {
						if r == reg {
							callerSavedFree[j] = true
							delete(regToValue, reg)
						}
					}
					for j, r := range calleeSavedPool {
						if r == reg {
							calleeSavedFree[j] = true
							delete(regToValue, reg)
						}
					}
				}
				if xmm, ok := a.XMMMap[id]; ok {
					for j, r := range xmmPool {
						if r == xmm {
							xmmFree[j] = true
							delete(xmmToValue, xmm)
						}
					}
				}
				if ymm, ok := a.YMMMap[id]; ok {
					for j, r := range ymmPool {
						if r == ymm {
							ymmFree[j] = true
							delete(ymmToValue, ymm)
						}
					}
				}
			}
		}
	}

	// Helper to allocate a caller-saved register
	allocCallerSaved := func() (Reg, bool) {
		for i, free := range callerSavedFree {
			if free {
				callerSavedFree[i] = false
				return callerSavedPool[i], true
			}
		}
		return 0, false
	}

	// Helper to allocate a callee-saved register
	allocCalleeSaved := func() (Reg, bool) {
		for i, free := range calleeSavedFree {
			if free {
				calleeSavedFree[i] = false
				calleeSavedEverUsed[i] = true
				return calleeSavedPool[i], true
			}
		}
		return 0, false
	}

	// Helper to allocate an XMM register
	allocXMM := func() (XMMReg, bool) {
		for i, free := range xmmFree {
			if free {
				xmmFree[i] = false
				return xmmPool[i], true
			}
		}
		return 0, false
	}

	// Helper to allocate a YMM register
	allocYMM := func() (YMMReg, bool) {
		for i, free := range ymmFree {
			if free {
				ymmFree[i] = false
				return ymmPool[i], true
			}
		}
		return 0, false
	}

	// Map parameters first
	for i := 0; i < len(fn.Params); i++ {
		paramType := fn.Params[i].Type
		if isFloatType(paramType) {
			// Float parameters come in XMM registers
			if xmm, ok := allocXMM(); ok {
				a.XMMMap[i] = xmm
				xmmToValue[xmm] = i
			} else {
				a.StackSlots[i] = int32(-(i + 1) * 8)
			}
		} else {
			paramReg := SysVIntArgs[i]
			if hasCall && liveAcrossCall[i] {
				if reg, ok := allocCalleeSaved(); ok {
					a.Map[i] = reg
					regToValue[reg] = i
				} else {
					a.StackSlots[i] = int32(-(i + 1) * 8)
				}
			} else {
				if reg, ok := allocCallerSaved(); ok {
					a.Map[i] = reg
					regToValue[reg] = i
				} else {
					a.Map[i] = paramReg
				}
			}
		}
	}

	// Allocate instruction results
	stackOff := int32(-8)
	for i, in := range fn.Body {
		// Free registers for values that died in previous instruction
		freeDeadRegs(i)

		if _, seen := a.Map[in.Result.ID]; seen {
			continue
		}
		if _, seen := a.XMMMap[in.Result.ID]; seen {
			continue
		}
		if _, seen := a.YMMMap[in.Result.ID]; seen {
			continue
		}
		if _, seen := a.StackSlots[in.Result.ID]; seen {
			continue
		}

		// Live-interval coalescing for OpCopy: if the source value's last
		// use is exactly this OpCopy instruction, the source's live range
		// ends here and the destination's begins. Assigning both to the
		// same physical register turns the OpCopy into a no-op (the isel
		// already short-circuits `mov r, r`). We bypass the normal
		// allocation path and remove the source from lastUse so a later
		// freeDeadRegs call doesn't reclaim a register that's still live
		// under its new SSA name.
		if in.Op == mir.OpCopy && len(in.Args) == 1 {
			srcID := in.Args[0].ID
			if lu, ok := lastUse[srcID]; ok && lu == i {
				if reg, ok := a.Map[srcID]; ok {
					a.Map[in.Result.ID] = reg
					regToValue[reg] = in.Result.ID
					delete(lastUse, srcID)
					continue
				}
				if xmm, ok := a.XMMMap[srcID]; ok {
					a.XMMMap[in.Result.ID] = xmm
					xmmToValue[xmm] = in.Result.ID
					delete(lastUse, srcID)
					continue
				}
				if ymm, ok := a.YMMMap[srcID]; ok {
					a.YMMMap[in.Result.ID] = ymm
					ymmToValue[ymm] = in.Result.ID
					delete(lastUse, srcID)
					continue
				}
			}
		}

		// Check if this is a Vec8F32 value (AVX2 packed floats)
		if isVec8F32Type(in.Result.Type) {
			if ymm, ok := allocYMM(); ok {
				a.YMMMap[in.Result.ID] = ymm
				ymmToValue[ymm] = in.Result.ID
			} else {
				a.StackSlots[in.Result.ID] = stackOff
				stackOff -= 32 // YMM registers are 32 bytes
			}
		} else if isVec4F32Type(in.Result.Type) {
			// Vec4F32 uses XMM registers (same as scalar floats)
			if xmm, ok := allocXMM(); ok {
				a.XMMMap[in.Result.ID] = xmm
				xmmToValue[xmm] = in.Result.ID
			} else {
				a.StackSlots[in.Result.ID] = stackOff
				stackOff -= 16 // XMM registers are 16 bytes
			}
		} else if isFloatType(in.Result.Type) {
			if xmm, ok := allocXMM(); ok {
				a.XMMMap[in.Result.ID] = xmm
				xmmToValue[xmm] = in.Result.ID
			} else {
				a.StackSlots[in.Result.ID] = stackOff
				stackOff -= 8
			}
		} else if hasCall && liveAcrossCall[in.Result.ID] {
			if reg, ok := allocCalleeSaved(); ok {
				a.Map[in.Result.ID] = reg
				regToValue[reg] = in.Result.ID
			} else {
				a.StackSlots[in.Result.ID] = stackOff
				stackOff -= 8
			}
		} else {
			if reg, ok := allocCallerSaved(); ok {
				a.Map[in.Result.ID] = reg
				regToValue[reg] = in.Result.ID
			} else if reg, ok := allocCalleeSaved(); ok {
				a.Map[in.Result.ID] = reg
				regToValue[reg] = in.Result.ID
			} else {
				a.StackSlots[in.Result.ID] = stackOff
				stackOff -= 8
			}
		}
	}

	// Record which callee-saved registers we ever used
	for i, used := range calleeSavedEverUsed {
		if used {
			a.NeedsSave = append(a.NeedsSave, calleeSavedPool[i])
		}
	}

	// Reserve save/restore slots for SIMD values live across calls.
	// All XMM/YMM regs are caller-saved on System V AMD64; we must
	// stash any cross-call SIMD value before the call and reload it
	// after. Slot per value is reused across calls.
	a.CallSaves = make([][]CallSaveEntry, len(callPositions))
	if hasCall {
		// Per-value slot, allocated lazily on first cross-call use.
		saveSlot := map[int]int32{}
		saveKind := map[int]CallSaveKind{}
		var saveOff int32 // grows from 0 up to SaveAreaSize

		alignTo := func(off, align int32) int32 {
			r := off % align
			if r != 0 {
				off += align - r
			}
			return off
		}

		// Walk values in increasing ID for deterministic layout.
		// We need (a) which values are XMM/YMM and (b) which calls
		// each crosses. Build the list of cross-call values first,
		// then assign slots, then build per-call entry lists.
		type valKind struct {
			id   int
			kind CallSaveKind
			xmm  XMMReg
			ymm  YMMReg
		}
		simdValues := []valKind{}
		for id := range a.XMMMap {
			if liveAcrossCall[id] {
				simdValues = append(simdValues, valKind{id: id, kind: SaveXMM, xmm: a.XMMMap[id]})
			}
		}
		for id := range a.YMMMap {
			if liveAcrossCall[id] {
				simdValues = append(simdValues, valKind{id: id, kind: SaveYMM, ymm: a.YMMMap[id]})
			}
		}
		// Sort by id for stable layout.
		for i := 1; i < len(simdValues); i++ {
			for j := i; j > 0 && simdValues[j-1].id > simdValues[j].id; j-- {
				simdValues[j-1], simdValues[j] = simdValues[j], simdValues[j-1]
			}
		}
		// YMM first (32-byte alignment cheapest at offset 0).
		// Reorder: YMM entries before XMM entries.
		ymmFirst := make([]valKind, 0, len(simdValues))
		for _, v := range simdValues {
			if v.kind == SaveYMM {
				ymmFirst = append(ymmFirst, v)
			}
		}
		for _, v := range simdValues {
			if v.kind == SaveXMM {
				ymmFirst = append(ymmFirst, v)
			}
		}
		simdValues = ymmFirst

		for _, v := range simdValues {
			switch v.kind {
			case SaveYMM:
				saveOff = alignTo(saveOff, 32)
				saveSlot[v.id] = saveOff
				saveKind[v.id] = SaveYMM
				saveOff += 32
			case SaveXMM:
				saveOff = alignTo(saveOff, 16)
				saveSlot[v.id] = saveOff
				saveKind[v.id] = SaveXMM
				saveOff += 16
			}
		}
		// Round to 32 to keep RSP nicely aligned.
		saveOff = alignTo(saveOff, 32)
		a.SaveAreaSize = saveOff

		// Build per-call entry lists.
		definedAt := map[int]int{}
		for i := 0; i < len(fn.Params); i++ {
			definedAt[i] = -1
		}
		for i, in := range fn.Body {
			definedAt[in.Result.ID] = i
		}
		for ci, callIdx := range callPositions {
			var list []CallSaveEntry
			for _, v := range simdValues {
				if definedAt[v.id] < callIdx && lastUse[v.id] > callIdx {
					e := CallSaveEntry{
						Slot: saveSlot[v.id],
						Kind: v.kind,
						XMM:  v.xmm,
						YMM:  v.ymm,
					}
					list = append(list, e)
				}
			}
			a.CallSaves[ci] = list
		}
	}

	// Calculate stack size
	if stackOff < -8 {
		a.StackSize = -stackOff
	}
	a.StackSize += int32(len(a.NeedsSave) * 8)

	// Ensure 16-byte alignment
	if a.StackSize%16 != 0 {
		a.StackSize += 16 - (a.StackSize % 16)
	}

	return a
}

func (a *Allocation) GetReg(id int) (Reg, bool) {
	r, ok := a.Map[id]
	return r, ok
}

func (a *Allocation) GetXMM(id int) (XMMReg, bool) {
	r, ok := a.XMMMap[id]
	return r, ok
}

func (a *Allocation) GetStack(id int) (int32, bool) {
	off, ok := a.StackSlots[id]
	return off, ok
}

func (a *Allocation) GetYMM(id int) (YMMReg, bool) {
	r, ok := a.YMMMap[id]
	return r, ok
}

// isVec8F32Type checks if a type should use YMM registers.
func isVec8F32Type(t *types.Type) bool {
	if t == nil {
		return false
	}
	return t.K == types.KVec8F32
}

// isVec4F32Type checks if a type should use XMM registers for packed ops.
func isVec4F32Type(t *types.Type) bool {
	if t == nil {
		return false
	}
	return t.K == types.KVec4F32
}
