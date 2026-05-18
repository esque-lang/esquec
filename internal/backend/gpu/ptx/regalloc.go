// Package ptx implements register allocation for PTX kernels.
package ptx

import (
	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/types"
)

// RegAlloc performs SSA-based register allocation for PTX.
type RegAlloc struct {
	// Value to register mapping per type
	allocB32 map[int]int
	allocB64 map[int]int
	allocF32 map[int]int
	allocF64 map[int]int
	allocPred map[int]int

	// Next available register per type
	nextB32 int
	nextB64 int
	nextF32 int
	nextF64 int
	nextPred int

	// Liveness information: for each value, at which instruction it dies
	lastUse map[int]int

	// Free register pools (for reuse after values die)
	freeB32 []int
	freeB64 []int
	freeF32 []int
	freeF64 []int
	freePred []int

	// Spill slots for values that don't fit in registers
	spillSlots map[int]int32
	nextSpill  int32

	// Register limits (PTX has generous limits but we should track)
	maxRegsPerType int
}

// NewRegAlloc creates a new register allocator.
func NewRegAlloc() *RegAlloc {
	return &RegAlloc{
		allocB32:  make(map[int]int),
		allocB64:  make(map[int]int),
		allocF32:  make(map[int]int),
		allocF64:  make(map[int]int),
		allocPred: make(map[int]int),
		lastUse:   make(map[int]int),
		spillSlots: make(map[int]int32),
		maxRegsPerType: 255, // PTX allows 255 registers per type
	}
}

// Allocate performs register allocation for a MIR function.
// Returns the allocation or an error.
func (ra *RegAlloc) Allocate(f *mir.Func) (*Allocation, error) {
	// Phase 1: Compute liveness (last use of each value)
	ra.computeLiveness(f.Body)

	// Phase 2: Allocate registers in program order
	for i, inst := range f.Body {
		// Free registers for values that die at this instruction
		ra.freeDeadValues(i)

		// Allocate register for result (if any)
		if inst.Result.ID > 0 {
			ra.allocateValue(inst.Result.ID, inst.Result.Type)
		}
	}

	// Build the allocation result
	return &Allocation{
		B32:       copyMap(ra.allocB32),
		B64:       copyMap(ra.allocB64),
		F32:       copyMap(ra.allocF32),
		F64:       copyMap(ra.allocF64),
		Pred:      copyMap(ra.allocPred),
		SpillSlots: copySpillMap(ra.spillSlots),
		RegCountB32: ra.nextB32,
		RegCountB64: ra.nextB64,
		RegCountF32: ra.nextF32,
		RegCountF64: ra.nextF64,
		RegCountPred: ra.nextPred,
	}, nil
}

// Allocation holds the result of register allocation.
type Allocation struct {
	B32  map[int]int // Value ID -> B32 register number
	B64  map[int]int // Value ID -> B64 register number
	F32  map[int]int // Value ID -> F32 register number
	F64  map[int]int // Value ID -> F64 register number
	Pred map[int]int // Value ID -> predicate register number

	SpillSlots map[int]int32 // Value ID -> spill slot offset

	// Total registers needed per type
	RegCountB32 int
	RegCountB64 int
	RegCountF32 int
	RegCountF64 int
	RegCountPred int
}

// GetReg returns the register number for a value, or -1 if spilled.
func (a *Allocation) GetReg(id int, t *types.Type) int {
	rt := regKind(t)
	switch rt {
	case regKindB32:
		if r, ok := a.B32[id]; ok {
			return r
		}
	case regKindB64:
		if r, ok := a.B64[id]; ok {
			return r
		}
	case regKindF32:
		if r, ok := a.F32[id]; ok {
			return r
		}
	case regKindF64:
		if r, ok := a.F64[id]; ok {
			return r
		}
	case regKindPred:
		if r, ok := a.Pred[id]; ok {
			return r
		}
	}
	return -1
}

// IsSpilled returns true if the value is spilled to memory.
func (a *Allocation) IsSpilled(id int) bool {
	_, ok := a.SpillSlots[id]
	return ok
}

// GetSpillSlot returns the spill slot offset for a value.
func (a *Allocation) GetSpillSlot(id int) (int32, bool) {
	slot, ok := a.SpillSlots[id]
	return slot, ok
}

// computeLiveness computes the last use of each value.
func (ra *RegAlloc) computeLiveness(body []mir.Inst) {
	for i, inst := range body {
		// Record last use of each argument
		for _, arg := range inst.Args {
			if arg.ID > 0 {
				ra.lastUse[arg.ID] = i
			}
		}
	}
}

// freeDeadValues returns registers for values that die at instruction i.
func (ra *RegAlloc) freeDeadValues(i int) {
	// Find values that die at this instruction
	for id, lastUseInst := range ra.lastUse {
		if lastUseInst < i {
			// Value is dead, free its register
			if reg, ok := ra.allocB32[id]; ok {
				ra.freeB32 = append(ra.freeB32, reg)
				delete(ra.allocB32, id)
			}
			if reg, ok := ra.allocB64[id]; ok {
				ra.freeB64 = append(ra.freeB64, reg)
				delete(ra.allocB64, id)
			}
			if reg, ok := ra.allocF32[id]; ok {
				ra.freeF32 = append(ra.freeF32, reg)
				delete(ra.allocF32, id)
			}
			if reg, ok := ra.allocF64[id]; ok {
				ra.freeF64 = append(ra.freeF64, reg)
				delete(ra.allocF64, id)
			}
			if reg, ok := ra.allocPred[id]; ok {
				ra.freePred = append(ra.freePred, reg)
				delete(ra.allocPred, id)
			}
			delete(ra.lastUse, id)
		}
	}
}

// allocateValue assigns a register or spill slot to a value.
func (ra *RegAlloc) allocateValue(id int, t *types.Type) {
	rt := regKind(t)

	switch rt {
	case regKindB32:
		if len(ra.freeB32) > 0 {
			reg := ra.freeB32[len(ra.freeB32)-1]
			ra.freeB32 = ra.freeB32[:len(ra.freeB32)-1]
			ra.allocB32[id] = reg
		} else if ra.nextB32 < ra.maxRegsPerType {
			ra.allocB32[id] = ra.nextB32
			ra.nextB32++
		} else {
			ra.spill(id)
		}

	case regKindB64:
		if len(ra.freeB64) > 0 {
			reg := ra.freeB64[len(ra.freeB64)-1]
			ra.freeB64 = ra.freeB64[:len(ra.freeB64)-1]
			ra.allocB64[id] = reg
		} else if ra.nextB64 < ra.maxRegsPerType {
			ra.allocB64[id] = ra.nextB64
			ra.nextB64++
		} else {
			ra.spill(id)
		}

	case regKindF32:
		if len(ra.freeF32) > 0 {
			reg := ra.freeF32[len(ra.freeF32)-1]
			ra.freeF32 = ra.freeF32[:len(ra.freeF32)-1]
			ra.allocF32[id] = reg
		} else if ra.nextF32 < ra.maxRegsPerType {
			ra.allocF32[id] = ra.nextF32
			ra.nextF32++
		} else {
			ra.spill(id)
		}

	case regKindF64:
		if len(ra.freeF64) > 0 {
			reg := ra.freeF64[len(ra.freeF64)-1]
			ra.freeF64 = ra.freeF64[:len(ra.freeF64)-1]
			ra.allocF64[id] = reg
		} else if ra.nextF64 < ra.maxRegsPerType {
			ra.allocF64[id] = ra.nextF64
			ra.nextF64++
		} else {
			ra.spill(id)
		}

	case regKindPred:
		if len(ra.freePred) > 0 {
			reg := ra.freePred[len(ra.freePred)-1]
			ra.freePred = ra.freePred[:len(ra.freePred)-1]
			ra.allocPred[id] = reg
		} else if ra.nextPred < ra.maxRegsPerType {
			ra.allocPred[id] = ra.nextPred
			ra.nextPred++
		} else {
			ra.spill(id)
		}
	}
}

// spill allocates a spill slot for a value.
func (ra *RegAlloc) spill(id int) {
	ra.spillSlots[id] = ra.nextSpill
	ra.nextSpill += 8 // 8-byte alignment for all values
}

type regKindT int

const (
	regKindB32 regKindT = iota
	regKindB64
	regKindF32
	regKindF64
	regKindPred
)

// regKind returns the register kind for a type.
func regKind(t *types.Type) regKindT {
	if t == nil {
		return regKindB64
	}
	switch t.K {
	case types.KBool:
		return regKindPred
	case types.KI8, types.KU8, types.KI16, types.KU16, types.KI32, types.KU32:
		return regKindB32
	case types.KI64, types.KU64:
		return regKindB64
	case types.KF32:
		return regKindF32
	case types.KF64:
		return regKindF64
	}
	return regKindB64
}

func copyMap(m map[int]int) map[int]int {
	c := make(map[int]int, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

func copySpillMap(m map[int]int32) map[int]int32 {
	c := make(map[int]int32, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}
