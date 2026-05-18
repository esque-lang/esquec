// Package ptx implements memory optimizations for PTX kernels.
package ptx

import (
	"fmt"
)

// MemoryConfig holds memory optimization settings.
type MemoryConfig struct {
	// Shared memory settings
	SharedMemSize int64 // Total shared memory per block (bytes)
	UseTiling     bool  // Enable memory tiling

	// Tiling parameters
	TileM int // Tile size for M dimension
	TileN int // Tile size for N dimension
	TileK int // Tile size for K dimension

	// Coalescing settings
	CoalesceAccess bool // Enable coalesced memory access patterns
	Alignment      int  // Memory alignment (typically 32 or 128 bytes)
}

// DefaultMemoryConfig returns sensible defaults.
func DefaultMemoryConfig() *MemoryConfig {
	return &MemoryConfig{
		SharedMemSize:  48 * 1024, // 48KB per block (typical SM limit)
		UseTiling:      true,
		TileM:          32,
		TileN:          32,
		TileK:          8,
		CoalesceAccess: true,
		Alignment:      128, // 128-byte aligned for optimal coalescing
	}
}

// SharedAlloc represents a shared memory allocation.
type SharedAlloc struct {
	Name      string // Variable name
	Type      RegType
	Count     int64  // Number of elements
	Alignment int64
	Offset    int64 // Offset in shared memory segment
}

// SharedMemManager manages shared memory allocations.
type SharedMemManager struct {
	config     *MemoryConfig
	allocs     []*SharedAlloc
	nextOffset int64
}

// NewSharedMemManager creates a new shared memory manager.
func NewSharedMemManager(config *MemoryConfig) *SharedMemManager {
	if config == nil {
		config = DefaultMemoryConfig()
	}
	return &SharedMemManager{
		config: config,
	}
}

// Alloc allocates shared memory.
func (m *SharedMemManager) Alloc(name string, elemType RegType, count int64) (*SharedAlloc, error) {
	elemSize := regTypeSize(elemType)
	alignment := int64(m.config.Alignment)

	// Align the offset
	if m.nextOffset%alignment != 0 {
		m.nextOffset = (m.nextOffset + alignment - 1) / alignment * alignment
	}

	totalSize := elemSize * count
	if m.nextOffset+totalSize > m.config.SharedMemSize {
		return nil, fmt.Errorf("shared memory exhausted: need %d bytes, have %d",
			totalSize, m.config.SharedMemSize-m.nextOffset)
	}

	alloc := &SharedAlloc{
		Name:      name,
		Type:      elemType,
		Count:     count,
		Alignment: alignment,
		Offset:    m.nextOffset,
	}

	m.allocs = append(m.allocs, alloc)
	m.nextOffset += totalSize

	return alloc, nil
}

// TotalUsed returns the total shared memory used.
func (m *SharedMemManager) TotalUsed() int64 {
	return m.nextOffset
}

// EmitDeclarations generates PTX shared memory declarations.
func (m *SharedMemManager) EmitDeclarations() []Inst {
	var insts []Inst
	// Shared memory is declared as a single block at kernel level
	// Individual allocations are accessed via offsets
	return insts
}

func regTypeSize(rt RegType) int64 {
	switch rt {
	case RegPred:
		return 1
	case RegB8:
		return 1
	case RegB16:
		return 2
	case RegB32, RegF32:
		return 4
	case RegB64, RegF64:
		return 8
	}
	return 8
}

// TiledMatMul generates instructions for tiled matrix multiplication.
// This uses shared memory to reduce global memory bandwidth.
func TiledMatMul(m *SharedMemManager, aBase, bBase, cBase string, M, N, K int64) ([]Inst, error) {
	config := m.config
	tileM := int64(config.TileM)
	tileN := int64(config.TileN)
	tileK := int64(config.TileK)

	// Allocate shared memory for tiles
	tileA, err := m.Alloc("smem_A", RegF32, tileM*tileK)
	if err != nil {
		return nil, fmt.Errorf("allocating tile A: %w", err)
	}
	tileB, err := m.Alloc("smem_B", RegF32, tileK*tileN)
	if err != nil {
		return nil, fmt.Errorf("allocating tile B: %w", err)
	}

	var insts []Inst

	// Thread indexing
	insts = append(insts,
		// Get thread ID
		Inst{
			Op:   OpTid,
			Type: RegB32,
			Dest: Operand{Kind: OpndReg, Reg: "tidx"},
		},
		// Get block ID
		Inst{
			Op:   OpCtaid,
			Type: RegB32,
			Dest: Operand{Kind: OpndReg, Reg: "bidx"},
		},
	)

	// Compute row/col indices
	// row = bidx.x * tileM + tidx.x / tileN
	// col = bidy.y * tileN + tidx.x % tileN
	insts = append(insts,
		Inst{
			Op:   OpMov,
			Type: RegB32,
			Dest: Operand{Kind: OpndReg, Reg: "tileM"},
			Src:  []Operand{{Kind: OpndImm, Imm: tileM}},
		},
		Inst{
			Op:   OpMov,
			Type: RegB32,
			Dest: Operand{Kind: OpndReg, Reg: "tileN"},
			Src:  []Operand{{Kind: OpndImm, Imm: tileN}},
		},
		Inst{
			Op:   OpMov,
			Type: RegB32,
			Dest: Operand{Kind: OpndReg, Reg: "tileK"},
			Src:  []Operand{{Kind: OpndImm, Imm: tileK}},
		},
	)

	// Accumulator initialization
	insts = append(insts,
		Inst{
			Op:   OpMov,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: "acc"},
			Src:  []Operand{{Kind: OpndImmF, ImmF: 0.0}},
		},
	)

	// Loop label for K tiles
	insts = append(insts,
		Inst{Op: OpLabel, Label: "k_loop"},
	)

	// Load tile from A into shared memory
	insts = append(insts,
		// Load A[row, k_offset + local_k] -> smem_A[local_row, local_k]
		Inst{
			Op:         OpLd,
			Type:       RegF32,
			StateSpace: SSGlobal,
			Dest:       Operand{Kind: OpndReg, Reg: "a_val"},
			Src:        []Operand{{Kind: OpndAddr, Reg: aBase, Offset: 0}},
		},
		Inst{
			Op:         OpSt,
			Type:       RegF32,
			StateSpace: SSShared,
			Dest:       Operand{Kind: OpndAddr, Reg: fmt.Sprintf("smem+%d", tileA.Offset)},
			Src:        []Operand{{Kind: OpndReg, Reg: "a_val"}},
		},
	)

	// Load tile from B into shared memory
	insts = append(insts,
		Inst{
			Op:         OpLd,
			Type:       RegF32,
			StateSpace: SSGlobal,
			Dest:       Operand{Kind: OpndReg, Reg: "b_val"},
			Src:        []Operand{{Kind: OpndAddr, Reg: bBase, Offset: 0}},
		},
		Inst{
			Op:         OpSt,
			Type:       RegF32,
			StateSpace: SSShared,
			Dest:       Operand{Kind: OpndAddr, Reg: fmt.Sprintf("smem+%d", tileB.Offset)},
			Src:        []Operand{{Kind: OpndReg, Reg: "b_val"}},
		},
	)

	// Synchronize threads
	insts = append(insts, Inst{Op: OpBar})

	// Inner loop: compute partial products
	insts = append(insts,
		Inst{Op: OpLabel, Label: "inner_k"},
		// Load from shared memory
		Inst{
			Op:         OpLd,
			Type:       RegF32,
			StateSpace: SSShared,
			Dest:       Operand{Kind: OpndReg, Reg: "a_shared"},
			Src:        []Operand{{Kind: OpndAddr, Reg: fmt.Sprintf("smem+%d", tileA.Offset)}},
		},
		Inst{
			Op:         OpLd,
			Type:       RegF32,
			StateSpace: SSShared,
			Dest:       Operand{Kind: OpndReg, Reg: "b_shared"},
			Src:        []Operand{{Kind: OpndAddr, Reg: fmt.Sprintf("smem+%d", tileB.Offset)}},
		},
		// FMA: acc += a * b
		Inst{
			Op:   OpFMa,
			Type: RegF32,
			Dest: Operand{Kind: OpndReg, Reg: "acc"},
			Src: []Operand{
				{Kind: OpndReg, Reg: "a_shared"},
				{Kind: OpndReg, Reg: "b_shared"},
				{Kind: OpndReg, Reg: "acc"},
			},
		},
	)

	// Synchronize before loading next tile
	insts = append(insts, Inst{Op: OpBar})

	// K loop control
	insts = append(insts,
		// Update k counter and branch
		Inst{
			Op:   OpAdd,
			Type: RegB32,
			Dest: Operand{Kind: OpndReg, Reg: "k_idx"},
			Src: []Operand{
				{Kind: OpndReg, Reg: "k_idx"},
				{Kind: OpndImm, Imm: tileK},
			},
		},
		Inst{
			Op:    OpSetp,
			Type:  RegB32,
			CmpOp: CmpLt,
			Dest:  Operand{Kind: OpndReg, Reg: "p_k"},
			Src: []Operand{
				{Kind: OpndReg, Reg: "k_idx"},
				{Kind: OpndImm, Imm: K},
			},
		},
		Inst{
			Op:    OpBra,
			Pred:  "p_k",
			Label: "k_loop",
		},
	)

	// Store result to C
	insts = append(insts,
		Inst{
			Op:         OpSt,
			Type:       RegF32,
			StateSpace: SSGlobal,
			Dest:       Operand{Kind: OpndAddr, Reg: cBase, Offset: 0},
			Src:        []Operand{{Kind: OpndReg, Reg: "acc"}},
		},
	)

	return insts, nil
}

// CoalescedLoad generates instructions for coalesced memory loads.
// Coalesced access means consecutive threads access consecutive memory addresses.
func CoalescedLoad(baseReg string, offset int64, count int, elemType RegType) []Inst {
	var insts []Inst

	elemSize := regTypeSize(elemType)

	// Get thread ID for calculating per-thread offset
	insts = append(insts, Inst{
		Op:   OpTid,
		Type: RegB32,
		Dest: Operand{Kind: OpndReg, Reg: "tid_coal"},
	})

	// Calculate address: base + offset + tid * elemSize
	insts = append(insts, Inst{
		Op:   OpMov,
		Type: RegB64,
		Dest: Operand{Kind: OpndReg, Reg: "elem_size"},
		Src:  []Operand{{Kind: OpndImm, Imm: elemSize}},
	})
	insts = append(insts, Inst{
		Op:   OpMul,
		Type: RegB64,
		Dest: Operand{Kind: OpndReg, Reg: "tid_offset"},
		Src: []Operand{
			{Kind: OpndReg, Reg: "tid_coal"},
			{Kind: OpndReg, Reg: "elem_size"},
		},
	})
	insts = append(insts, Inst{
		Op:   OpAdd,
		Type: RegB64,
		Dest: Operand{Kind: OpndReg, Reg: "load_addr"},
		Src: []Operand{
			{Kind: OpndReg, Reg: baseReg},
			{Kind: OpndReg, Reg: "tid_offset"},
		},
	})

	// Perform the load
	insts = append(insts, Inst{
		Op:         OpLd,
		Type:       elemType,
		StateSpace: SSGlobal,
		Dest:       Operand{Kind: OpndReg, Reg: "val_coal"},
		Src:        []Operand{{Kind: OpndAddr, Reg: "load_addr", Offset: offset}},
	})

	return insts
}

// CoalescedStore generates instructions for coalesced memory stores.
func CoalescedStore(baseReg string, offset int64, valReg string, elemType RegType) []Inst {
	var insts []Inst

	elemSize := regTypeSize(elemType)

	// Get thread ID
	insts = append(insts, Inst{
		Op:   OpTid,
		Type: RegB32,
		Dest: Operand{Kind: OpndReg, Reg: "tid_coal_st"},
	})

	// Calculate address
	insts = append(insts, Inst{
		Op:   OpMov,
		Type: RegB64,
		Dest: Operand{Kind: OpndReg, Reg: "elem_size_st"},
		Src:  []Operand{{Kind: OpndImm, Imm: elemSize}},
	})
	insts = append(insts, Inst{
		Op:   OpMul,
		Type: RegB64,
		Dest: Operand{Kind: OpndReg, Reg: "tid_offset_st"},
		Src: []Operand{
			{Kind: OpndReg, Reg: "tid_coal_st"},
			{Kind: OpndReg, Reg: "elem_size_st"},
		},
	})
	insts = append(insts, Inst{
		Op:   OpAdd,
		Type: RegB64,
		Dest: Operand{Kind: OpndReg, Reg: "store_addr"},
		Src: []Operand{
			{Kind: OpndReg, Reg: baseReg},
			{Kind: OpndReg, Reg: "tid_offset_st"},
		},
	})

	// Store
	insts = append(insts, Inst{
		Op:         OpSt,
		Type:       elemType,
		StateSpace: SSGlobal,
		Dest:       Operand{Kind: OpndAddr, Reg: "store_addr", Offset: offset},
		Src:        []Operand{{Kind: OpndReg, Reg: valReg}},
	})

	return insts
}

// VectorLoad generates PTX vector load instructions (ld.v2, ld.v4).
// Vector loads load multiple elements in a single instruction.
func VectorLoad(baseReg string, offset int64, elemType RegType, count int) []Inst {
	var insts []Inst

	switch count {
	case 2:
		// ld.global.v2.f32 {r0, r1}, [addr]
		insts = append(insts, Inst{
			Op:         OpLd,
			Type:       elemType,
			StateSpace: SSGlobal,
			Dest:       Operand{Kind: OpndReg, Reg: "vec2"},
			Src:        []Operand{{Kind: OpndAddr, Reg: baseReg, Offset: offset}},
		})
	case 4:
		// ld.global.v4.f32 {r0, r1, r2, r3}, [addr]
		insts = append(insts, Inst{
			Op:         OpLd,
			Type:       elemType,
			StateSpace: SSGlobal,
			Dest:       Operand{Kind: OpndReg, Reg: "vec4"},
			Src:        []Operand{{Kind: OpndAddr, Reg: baseReg, Offset: offset}},
		})
	default:
		// Fall back to scalar loads
		for i := 0; i < count; i++ {
			insts = append(insts, Inst{
				Op:         OpLd,
				Type:       elemType,
				StateSpace: SSGlobal,
				Dest:       Operand{Kind: OpndReg, Reg: fmt.Sprintf("elem%d", i)},
				Src:        []Operand{{Kind: OpndAddr, Reg: baseReg, Offset: offset + int64(i)*regTypeSize(elemType)}},
			})
		}
	}

	return insts
}
