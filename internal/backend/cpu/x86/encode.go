package x86

import (
	"encoding/binary"
	"fmt"
)

// Encoder produces a flat byte stream of x86-64 machine code, plus a
// list of symbol references that need relocations applied later.
type Encoder struct {
	Code []byte
	Rels []Reloc
}

// Reloc is a single relocation request the encoder records when it
// emits a call/jump to a named symbol whose address is not yet known.
type Reloc struct {
	Offset int    // byte offset within Code where the displacement lives
	Symbol string // target symbol name
	Kind   RelKind
	Addend int64
}

type RelKind int

const (
	// RelPC32 — 32-bit PC-relative (R_X86_64_PC32 / PLT32)
	RelPC32 RelKind = iota
)

// rex emits a REX prefix when needed.
//
// w: operand-size (1 = 64-bit)
// r: extension of ModRM.reg
// x: extension of SIB.index
// b: extension of ModRM.rm or SIB.base or opcode reg
func (e *Encoder) rex(w, r, x, b uint8) {
	if w|r|x|b == 0 {
		return
	}
	e.Code = append(e.Code, 0x40|(w<<3)|(r<<2)|(x<<1)|b)
}

// modrm builds a ModR/M byte.
func modrm(mod, reg, rm uint8) byte {
	return (mod << 6) | ((reg & 7) << 3) | (rm & 7)
}

// emitMovImm32ToReg64 emits `mov <reg64>, <imm32>` (sign-extended).
//
// Encoding: REX.W + C7 /0 ib32  →  REX.W=1, opcode=C7, ModRM mod=11 reg=0 rm=reg
// (Equivalent to `mov r/m64, imm32` with the immediate sign-extended.)
func (e *Encoder) MovImm32ToReg64(r Reg, imm int32) {
	rb := uint8(0)
	if r >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0xC7)
	e.Code = append(e.Code, modrm(0b11, 0, uint8(r)))
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(imm))
	e.Code = append(e.Code, b[:]...)
}

// MovImm64ToReg64 emits `mov r64, imm64`  (REX.W + B8+rd io)
func (e *Encoder) MovImm64ToReg64(r Reg, imm uint64) {
	rb := uint8(0)
	if r >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0xB8+(uint8(r)&7))
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], imm)
	e.Code = append(e.Code, b[:]...)
}

// MovRegToReg64 emits `mov dst, src` (REX.W + 89 /r).
func (e *Encoder) MovRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x89)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// AddRegToReg64 emits `add dst, src` (REX.W + 01 /r).
func (e *Encoder) AddRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x01)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// SubRegFromReg64 emits `sub dst, src` (REX.W + 29 /r).
func (e *Encoder) SubRegFromReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x29)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// IMulRegByReg64 emits `imul dst, src` (REX.W + 0F AF /r).
func (e *Encoder) IMulRegByReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if src >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x0F, 0xAF)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// NegReg64 emits `neg r` (REX.W + F7 /3).
func (e *Encoder) NegReg64(r Reg) {
	rb := uint8(0)
	if r >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0xF7)
	e.Code = append(e.Code, modrm(0b11, 3, uint8(r)))
}

// IDIV divides RDX:RAX by src (signed). Result: quotient in RAX, remainder in RDX.
// Requires prior sign-extension of RAX into RDX via CQO.
// REX.W + F7 /7
func (e *Encoder) IDivReg64(src Reg) {
	rb := uint8(0)
	if src >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0xF7)
	e.Code = append(e.Code, modrm(0b11, 7, uint8(src)))
}

// DivReg64 divides RDX:RAX by src (unsigned). Quotient in RAX, remainder in RDX.
// Caller must zero RDX (via XOR RDX, RDX) before invoking.
// REX.W + F7 /6
func (e *Encoder) DivReg64(src Reg) {
	rb := uint8(0)
	if src >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0xF7)
	e.Code = append(e.Code, modrm(0b11, 6, uint8(src)))
}

// CQO sign-extends RAX into RDX:RAX (REX.W + 99).
func (e *Encoder) CQO() {
	e.rex(1, 0, 0, 0)
	e.Code = append(e.Code, 0x99)
}

// CmpRegToReg64 emits `cmp dst, src` (REX.W + 39 /r).
func (e *Encoder) CmpRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x39)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// SetCC sets dst to 1 or 0 based on condition code.
// Uses SETCC instruction (0F 9x /0).
// dst must be a register whose low 8-bit part is addressable.
func (e *Encoder) SetCC(cc CondCode, dst Reg) {
	rb := uint8(0)
	if dst >= R8 {
		rb = 1
	}
	// For R8-R15 or RSP-RDI we need REX prefix to access low byte
	if dst >= R8 || dst == RSP || dst == RBP || dst == RSI || dst == RDI {
		e.rex(0, 0, 0, rb)
	}
	e.Code = append(e.Code, 0x0F, 0x90+uint8(cc))
	e.Code = append(e.Code, modrm(0b11, 0, uint8(dst)))
}

// MovzxReg8ToReg64 zero-extends 8-bit to 64-bit (REX.W + 0F B6 /r).
func (e *Encoder) MovzxReg8ToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if src >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x0F, 0xB6)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MovsxReg8ToReg64 sign-extends 8-bit to 64-bit (REX.W + 0F BE /r).
// Used to canonicalise the low byte of an i8 value back into 64-bit
// register form after arithmetic that may have left junk in the upper
// bits or truncated the sign bit.
func (e *Encoder) MovsxReg8ToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if src >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x0F, 0xBE)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// CMov conditionally moves src to dst based on condition code.
// REX.W + 0F 4x /r
func (e *Encoder) CMov(cc CondCode, dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if src >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x0F, 0x40+uint8(cc))
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// XorRegToReg64 emits `xor dst, src` (REX.W + 31 /r).
func (e *Encoder) XorRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x31)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// AndRegToReg64 emits `and dst, src` (REX.W + 21 /r).
func (e *Encoder) AndRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x21)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// OrRegToReg64 emits `or dst, src` (REX.W + 09 /r).
func (e *Encoder) OrRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x09)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// TestRegToReg64 emits `test dst, src` (REX.W + 85 /r).
func (e *Encoder) TestRegToReg64(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x85)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// TestRegToReg32 emits `test r/m32, r32` (85 /r) without REX.W. The
// flag computation is on the low 32 bits, so SF reflects bit 31 of
// the source rather than bit 63 — useful when the high half of a
// register is known zero (e.g. after a 32-bit load) and we still want
// to test the original sign bit.
func (e *Encoder) TestRegToReg32(dst, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if dst >= R8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x85)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
}

// Push emits `push r64` (50+rd or REX + 50+rd).
func (e *Encoder) Push(r Reg) {
	if r >= R8 {
		e.Code = append(e.Code, 0x41) // REX.B
	}
	e.Code = append(e.Code, 0x50+(uint8(r)&7))
}

// Pop emits `pop r64` (58+rd or REX + 58+rd).
func (e *Encoder) Pop(r Reg) {
	if r >= R8 {
		e.Code = append(e.Code, 0x41) // REX.B
	}
	e.Code = append(e.Code, 0x58+(uint8(r)&7))
}

// SubImm32FromReg64 emits `sub r, imm32` (REX.W + 81 /5 id).
func (e *Encoder) SubImm32FromReg64(r Reg, imm int32) {
	rb := uint8(0)
	if r >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0x81)
	e.Code = append(e.Code, modrm(0b11, 5, uint8(r)))
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(imm))
	e.Code = append(e.Code, b[:]...)
}

// AddImm32ToReg64 emits `add r, imm32` (REX.W + 81 /0 id).
func (e *Encoder) AddImm32ToReg64(r Reg, imm int32) {
	rb := uint8(0)
	if r >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0x81)
	e.Code = append(e.Code, modrm(0b11, 0, uint8(r)))
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(imm))
	e.Code = append(e.Code, b[:]...)
}

// SyscallInsn emits `syscall` (0F 05).
func (e *Encoder) SyscallInsn() {
	e.Code = append(e.Code, 0x0F, 0x05)
}

// MovByteImmToMem emits `mov byte [base + disp32], imm8` (C6 /0 ib).
// Used by the runtime printer to write ASCII bytes into a stack buffer.
func (e *Encoder) MovByteImmToMem(base Reg, disp int32, imm byte) {
	rb := uint8(0)
	if base >= R8 {
		rb = 1
	}
	if rb != 0 {
		e.Code = append(e.Code, 0x40|rb) // REX.B
	}
	e.Code = append(e.Code, 0xC6)
	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, 0, 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, 0, uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
	e.Code = append(e.Code, imm)
}

// MovByteRegToMem emits `mov byte [base + disp32], src8` (88 /r).
// `src` is a 64-bit register; only the low 8 bits (e.g. AL/DL) are stored.
// AH/CH/DH/BH-style high-byte access is intentionally not supported.
func (e *Encoder) MovByteRegToMem(base Reg, disp int32, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	// SPL/BPL/SIL/DIL require a REX prefix to disambiguate from AH/CH/DH/BH.
	needRex := rr|rb != 0 || src == RSP || src == RBP || src == RSI || src == RDI
	if needRex {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x88)
	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// Ret emits the near-return `ret` (C3).
func (e *Encoder) Ret() {
	e.Code = append(e.Code, 0xC3)
}

// CondCode represents x86-64 condition codes.
type CondCode uint8

const (
	CCE  CondCode = 0x4 // equal (ZF=1)
	CCNE CondCode = 0x5 // not equal (ZF=0)
	CCL  CondCode = 0xC // less (signed, SF≠OF)
	CCLE CondCode = 0xE // less or equal (signed, ZF=1 or SF≠OF)
	CCG  CondCode = 0xF // greater (signed, ZF=0 and SF=OF)
	CCGE CondCode = 0xD // greater or equal (signed, SF=OF)
	CCB  CondCode = 0x2 // below (unsigned <, CF=1)
	CCBE CondCode = 0x6 // below or equal (unsigned <=, CF=1 or ZF=1)
	CCA  CondCode = 0x7 // above (unsigned >, CF=0 and ZF=0)
	CCAE CondCode = 0x3 // above or equal (unsigned >=, CF=0)
)

// CallSym emits `call <sym>` with a 32-bit PC-relative displacement
// and records a PC32 relocation against the symbol.
func (e *Encoder) CallSym(sym string) {
	e.Code = append(e.Code, 0xE8)
	off := len(e.Code)
	e.Code = append(e.Code, 0, 0, 0, 0)
	e.Rels = append(e.Rels, Reloc{
		Offset: off,
		Symbol: sym,
		Kind:   RelPC32,
		// PC-relative call to sym: addend -4 because the displacement is
		// computed from the end of the instruction.
		Addend: -4,
	})
}

// String formats the encoded bytes as space-separated hex (for tests/debug).
func (e *Encoder) Hex() string {
	out := make([]byte, 0, 3*len(e.Code))
	const hex = "0123456789abcdef"
	for i, b := range e.Code {
		if i > 0 {
			out = append(out, ' ')
		}
		out = append(out, hex[b>>4], hex[b&0xF])
	}
	return string(out)
}

// Sanity-check helper used in tests; panics if v doesn't fit in int32.
func mustInt32(v int64) int32 {
	if v > 0x7FFFFFFF || v < -0x80000000 {
		panic(fmt.Sprintf("immediate %d does not fit in int32", v))
	}
	return int32(v)
}

// ============================================================
// SSE Instructions for f32 scalar operations
// ============================================================

// MovSSRegToReg emits `movss dst, src` (F3 0F 10 /r).
// Copies a single f32 between XMM registers.
func (e *Encoder) MovSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x10)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MovSSLoadMem emits `movss xmm, [base + disp32]` (F3 0F 10 /r).
// Loads a single f32 from memory into an XMM register.
func (e *Encoder) MovSSLoadMem(dst XMMReg, base Reg, disp int32) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x10)

	// Addressing mode: [base + disp32]
	if base == RSP || base == R12 {
		// RSP/R12 needs SIB byte
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4)) // mod=10, rm=100 (SIB)
		e.Code = append(e.Code, 0x24)                       // SIB: scale=0, index=RSP(none), base=RSP
	} else if base == RBP || base == R13 {
		// RBP/R13 always needs displacement
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base))) // mod=10
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base))) // mod=10
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovSSStoreMem emits `movss [base + disp32], xmm` (F3 0F 11 /r).
// Stores a single f32 from an XMM register to memory.
func (e *Encoder) MovSSStoreMem(base Reg, disp int32, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= XMM8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x11)

	// Addressing mode: [base + disp32]
	if base == RSP || base == R12 {
		// RSP/R12 needs SIB byte
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4)) // mod=10, rm=100 (SIB)
		e.Code = append(e.Code, 0x24)                       // SIB: scale=0, index=RSP(none), base=RSP
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base))) // mod=10
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// AddSSRegToReg emits `addss dst, src` (F3 0F 58 /r).
// Adds two f32 scalars.
func (e *Encoder) AddSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x58)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// SubSSRegToReg emits `subss dst, src` (F3 0F 5C /r).
// Subtracts two f32 scalars.
func (e *Encoder) SubSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x5C)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MulSSRegToReg emits `mulss dst, src` (F3 0F 59 /r).
// Multiplies two f32 scalars.
func (e *Encoder) MulSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x59)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// DivSSRegToReg emits `divss dst, src` (F3 0F 5E /r).
// Divides two f32 scalars.
func (e *Encoder) DivSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x5E)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// ============================================================
// SSE2 Instructions for f64 scalar operations.
//
// Encoded identically to the SSE f32 forms above except for the
// mandatory prefix: SS uses F3, SD uses F2. Same opcodes for the
// arithmetic family (58/59/5C/5E), same opcodes for MOVSD load/store
// (10/11). REX handling is identical.
// ============================================================

// MovSDRegToReg emits `movsd dst, src` (F2 0F 10 /r).
// Copies a single f64 between XMM registers.
func (e *Encoder) MovSDRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x10)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MovSDLoadMem emits `movsd xmm, [base + disp32]` (F2 0F 10 /r).
func (e *Encoder) MovSDLoadMem(dst XMMReg, base Reg, disp int32) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x10)
	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovSDStoreMem emits `movsd [base + disp32], xmm` (F2 0F 11 /r).
func (e *Encoder) MovSDStoreMem(base Reg, disp int32, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= XMM8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x11)
	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// AddSDRegToReg emits `addsd dst, src` (F2 0F 58 /r).
func (e *Encoder) AddSDRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x58)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// SubSDRegToReg emits `subsd dst, src` (F2 0F 5C /r).
func (e *Encoder) SubSDRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x5C)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MulSDRegToReg emits `mulsd dst, src` (F2 0F 59 /r).
func (e *Encoder) MulSDRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x59)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// DivSDRegToReg emits `divsd dst, src` (F2 0F 5E /r).
func (e *Encoder) DivSDRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x5E)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MaxSSRegToReg emits `maxss dst, src` (F3 0F 5F /r).
// Returns max of two f32 scalars.
func (e *Encoder) MaxSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x5F)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MinSSRegToReg emits `minss dst, src` (F3 0F 5D /r).
// Returns min of two f32 scalars.
func (e *Encoder) MinSSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x5D)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// XorPSRegToReg emits `xorps dst, src` (0F 57 /r).
// XORs two XMM registers (used to zero a register: xorps xmm, xmm).
func (e *Encoder) XorPSRegToReg(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x57)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MovImm64ToMem stores a 64-bit immediate to `[base + disp]` as two
// 32-bit stores (low at disp, high at disp+4). x86 has no `mov r/m64,
// imm64`; the standard pattern is two `mov r/m32, imm32` writes. Used
// to materialise an f64 constant in a scratch slot before MOVSD.
func (e *Encoder) MovImm64ToMem(base Reg, disp int32, imm uint64) {
	e.MovImm32ToMem(base, disp, uint32(imm))
	e.MovImm32ToMem(base, disp+4, uint32(imm>>32))
}

// MovImm32ToMem emits `mov dword [base + disp32], imm32` (C7 /0 id).
// Stores a 32-bit immediate to memory (used for float constants).
func (e *Encoder) MovImm32ToMem(base Reg, disp int32, imm uint32) {
	rb := uint8(0)
	if base >= R8 {
		rb = 1
	}
	if rb != 0 {
		e.Code = append(e.Code, 0x40|rb) // REX.B
	}
	e.Code = append(e.Code, 0xC7)

	// Addressing mode: [base + disp32]
	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, 0, 4)) // mod=10, reg=0, rm=100 (SIB)
		e.Code = append(e.Code, 0x24)              // SIB: scale=0, index=RSP(none), base=RSP
	} else {
		e.Code = append(e.Code, modrm(0b10, 0, uint8(base))) // mod=10, reg=0
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
	binary.LittleEndian.PutUint32(b[:], imm)
	e.Code = append(e.Code, b[:]...)
}

// ============================================================
// Jump and Label Instructions for loop codegen
// ============================================================

// JmpRel8 emits `jmp rel8` (EB cb).
// Short jump with 8-bit displacement.
func (e *Encoder) JmpRel8(offset int8) {
	e.Code = append(e.Code, 0xEB, byte(offset))
}

// JmpRel32 emits `jmp rel32` (E9 cd).
// Near jump with 32-bit displacement.
func (e *Encoder) JmpRel32(offset int32) {
	e.Code = append(e.Code, 0xE9)
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(offset))
	e.Code = append(e.Code, b[:]...)
}

// JccRel32 emits a conditional jump with 32-bit displacement.
// Encoding: 0F 8x cd
func (e *Encoder) JccRel32(cc CondCode, offset int32) {
	e.Code = append(e.Code, 0x0F, 0x80+uint8(cc))
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(offset))
	e.Code = append(e.Code, b[:]...)
}

// CmpImm32ToReg64 emits `cmp r64, imm32` (REX.W + 81 /7 id).
func (e *Encoder) CmpImm32ToReg64(r Reg, imm int32) {
	rb := uint8(0)
	if r >= R8 {
		rb = 1
	}
	e.rex(1, 0, 0, rb)
	e.Code = append(e.Code, 0x81)
	e.Code = append(e.Code, modrm(0b11, 7, uint8(r)))
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(imm))
	e.Code = append(e.Code, b[:]...)
}

// ============================================================
// Memory Load/Store for GPR (for loop index, etc.)
// ============================================================

// MovLoadMem64 emits `mov r64, [base + disp32]` (REX.W + 8B /r).
func (e *Encoder) MovLoadMem64(dst Reg, base Reg, disp int32) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x8B)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovStoreMem64 emits `mov [base + disp32], r64` (REX.W + 89 /r).
func (e *Encoder) MovStoreMem64(base Reg, disp int32, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x89)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovLoadMem32 emits `mov r32, [base + disp32]` (8B /r) - loads 32 bits zero-extended to 64.
func (e *Encoder) MovLoadMem32(dst Reg, base Reg, disp int32) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x8B)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovStoreMem32 emits `mov [base + disp32], r32` (89 /r).
func (e *Encoder) MovStoreMem32(base Reg, disp int32, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= R8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x89)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// LeaRipSym emits `lea dst, [rip + sym]` (REX.W + 8D /r with mod=00,
// rm=101) and records a 32-bit PC-relative relocation against `sym`.
// The linker resolves the displacement to the symbol's address. Used
// to load addresses of .rodata blobs into a GPR.
func (e *Encoder) LeaRipSym(dst Reg, sym string) {
	rr := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	e.rex(1, rr, 0, 0)
	e.Code = append(e.Code, 0x8D)
	// ModRM: mod=00, reg=dst, rm=101 → [rip + disp32]
	e.Code = append(e.Code, modrm(0b00, uint8(dst), 0b101))
	off := len(e.Code)
	e.Code = append(e.Code, 0, 0, 0, 0)
	e.Rels = append(e.Rels, Reloc{
		Offset: off,
		Symbol: sym,
		Kind:   RelPC32,
		// Same convention as CallSym: PC32 addend -4 because the linker
		// computes S + A - P with P pointing at the disp32 location, but
		// the CPU's RIP at execution time is the end of the instruction.
		Addend: -4,
	})
}

// LEA emits `lea dst, [base + disp32]` (REX.W + 8D /r).
func (e *Encoder) LEA(dst Reg, base Reg, disp int32) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	e.rex(1, rr, 0, rb)
	e.Code = append(e.Code, 0x8D)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// ============================================================
// Conversion Instructions
// ============================================================

// CVTTSS2SI emits `cvttss2si r32, xmm` (F3 0F 2C /r).
// Converts f32 to i32 with truncation toward zero.
func (e *Encoder) CVTTSS2SI(dst Reg, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x2C)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// CVTSI2SS emits `cvtsi2ss xmm, r32` (F3 0F 2A /r).
// Converts i32 to f32.
func (e *Encoder) CVTSI2SS(dst XMMReg, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= R8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF3) // mandatory prefix
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x2A)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// CVTTSD2SI emits `cvttsd2si r32, xmm` (F2 0F 2C /r).
// Converts f64 to i32 with truncation toward zero.
func (e *Encoder) CVTTSD2SI(dst Reg, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= R8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x2C)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// CVTSI2SD emits `cvtsi2sd xmm, r32` (F2 0F 2A /r).
// Converts i32 to f64.
func (e *Encoder) CVTSI2SD(dst XMMReg, src Reg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= R8 {
		rb = 1
	}
	e.Code = append(e.Code, 0xF2)
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x2A)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// ============================================================
// SSE Packed Instructions for 128-bit (4 x f32) operations
// ============================================================

// MovUPSLoad128 emits `movups xmm, [base + disp32]` (0F 10 /r).
// Loads 4 packed f32 (unaligned) from memory into an XMM register.
func (e *Encoder) MovUPSLoad128(dst XMMReg, base Reg, disp int32) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x10)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovUPSStore128 emits `movups [base + disp32], xmm` (0F 11 /r).
// Stores 4 packed f32 from an XMM register to memory (unaligned).
func (e *Encoder) MovUPSStore128(base Reg, disp int32, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if src >= XMM8 {
		rr = 1
	}
	if base >= R8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x11)

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base)))
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(disp))
	e.Code = append(e.Code, b[:]...)
}

// MovUPSReg128 emits `movups dst, src` (0F 10 /r).
// Copies 4 packed f32 between XMM registers.
func (e *Encoder) MovUPSReg128(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x10)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// AddPSReg128 emits `addps dst, src` (0F 58 /r).
// Adds 4 packed f32.
func (e *Encoder) AddPSReg128(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x58)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// SubPSReg128 emits `subps dst, src` (0F 5C /r).
// Subtracts 4 packed f32.
func (e *Encoder) SubPSReg128(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x5C)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// MulPSReg128 emits `mulps dst, src` (0F 59 /r).
// Multiplies 4 packed f32.
func (e *Encoder) MulPSReg128(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x59)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// DivPSReg128 emits `divps dst, src` (0F 5E /r).
// Divides 4 packed f32.
func (e *Encoder) DivPSReg128(dst, src XMMReg) {
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb) // REX
	}
	e.Code = append(e.Code, 0x0F, 0x5E)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// ============================================================
// AVX2 Instructions for 256-bit packed f32 operations
// ============================================================

// vex2 emits a 2-byte VEX prefix: 0xC5 [R~vvvvLpp]
// Used when X=0, B=0, W=0, map=0x0F
// r: inverted REX.R (1 if reg < 8, 0 if reg >= 8)
// vvvv: source register number (0-15), will be inverted for encoding
// L: 0=128-bit (XMM), 1=256-bit (YMM)
// pp: 00=none, 01=66, 10=F3, 11=F2
func (e *Encoder) vex2(r, vvvv, L, pp uint8) {
	// vvvv is ones-complemented in the encoding
	byte2 := (r << 7) | ((^vvvv & 0xF) << 3) | (L << 2) | pp
	e.Code = append(e.Code, 0xC5, byte2)
}

// vex3 emits a 3-byte VEX prefix: 0xC4 [RXBmmmmm] [W~vvvvLpp]
// Required when using R8-R15 as rm operand or for non-0F escape maps.
// r, x, b: inverted REX bits (1 if corresponding field < 8)
// m: opcode map (1=0F, 2=0F38, 3=0F3A)
// w: operand size (0 for most AVX)
// vvvv: source register number (0-15), will be inverted for encoding
// L: 0=128-bit, 1=256-bit
// pp: 00=none, 01=66, 10=F3, 11=F2
func (e *Encoder) vex3(r, x, b, m, w, vvvv, L, pp uint8) {
	byte2 := (r << 7) | (x << 6) | (b << 5) | m
	// vvvv is ones-complemented in the encoding
	byte3 := (w << 7) | ((^vvvv & 0xF) << 3) | (L << 2) | pp
	e.Code = append(e.Code, 0xC4, byte2, byte3)
}

// VMovUPSLoad256 emits `vmovups ymm, [base + disp32]` (VEX.256.0F 10 /r).
// Loads 8 packed f32 (unaligned) from memory into a YMM register.
func (e *Encoder) VMovUPSLoad256(dst YMMReg, base Reg, disp int32) {
	// Determine if we need 3-byte VEX (when base >= R8)
	if base >= R8 {
		r := uint8(1) // dst < 8 inverted
		if dst >= 8 {
			r = 0
		}
		b := uint8(0) // base >= R8, inverted = 0
		e.vex3(r, 1, b, 1, 0, 0, 1, 0) // m=1 (0F), vvvv=0 (unused), L=1 (256-bit), pp=0
	} else {
		r := uint8(1)
		if dst >= 8 {
			r = 0
		}
		e.vex2(r, 0, 1, 0) // vvvv=0 (unused, will be inverted to 1111), L=1, pp=0
	}
	e.Code = append(e.Code, 0x10) // VMOVUPS opcode

	// ModRM and addressing
	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), 4))
		e.Code = append(e.Code, 0x24) // SIB: base=RSP/R12
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(dst), uint8(base)))
	}
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(disp))
	e.Code = append(e.Code, buf[:]...)
}

// VMovUPSStore256 emits `vmovups [base + disp32], ymm` (VEX.256.0F 11 /r).
// Stores 8 packed f32 from a YMM register to memory (unaligned).
func (e *Encoder) VMovUPSStore256(base Reg, disp int32, src YMMReg) {
	// Determine if we need 3-byte VEX (when base >= R8)
	if base >= R8 {
		r := uint8(1)
		if src >= 8 {
			r = 0
		}
		b := uint8(0) // base >= R8, inverted = 0
		e.vex3(r, 1, b, 1, 0, 0, 1, 0) // vvvv=0 (unused)
	} else {
		r := uint8(1)
		if src >= 8 {
			r = 0
		}
		e.vex2(r, 0, 1, 0) // vvvv=0 (unused)
	}
	e.Code = append(e.Code, 0x11) // VMOVUPS store opcode

	if base == RSP || base == R12 {
		e.Code = append(e.Code, modrm(0b10, uint8(src), 4))
		e.Code = append(e.Code, 0x24)
	} else {
		e.Code = append(e.Code, modrm(0b10, uint8(src), uint8(base)))
	}
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(disp))
	e.Code = append(e.Code, buf[:]...)
}

// VAddPS256 emits `vaddps dst, src1, src2` (VEX.256.0F 58 /r).
// Adds 8 packed f32: dst = src1 + src2.
func (e *Encoder) VAddPS256(dst, src1, src2 YMMReg) {
	e.emitVexArith256(0x58, dst, src1, src2)
}

// VSubPS256 emits `vsubps dst, src1, src2` (VEX.256.0F 5C /r).
// Subtracts 8 packed f32: dst = src1 - src2.
func (e *Encoder) VSubPS256(dst, src1, src2 YMMReg) {
	e.emitVexArith256(0x5C, dst, src1, src2)
}

// VMulPS256 emits `vmulps dst, src1, src2` (VEX.256.0F 59 /r).
// Multiplies 8 packed f32: dst = src1 * src2.
func (e *Encoder) VMulPS256(dst, src1, src2 YMMReg) {
	e.emitVexArith256(0x59, dst, src1, src2)
}

// VDivPS256 emits `vdivps dst, src1, src2` (VEX.256.0F 5E /r).
// Divides 8 packed f32: dst = src1 / src2.
func (e *Encoder) VDivPS256(dst, src1, src2 YMMReg) {
	e.emitVexArith256(0x5E, dst, src1, src2)
}

// emitVexArith256 emits a VEX-encoded 3-operand AVX instruction.
// opcode is the final opcode byte (e.g., 0x58 for VADDPS).
func (e *Encoder) emitVexArith256(opcode uint8, dst, src1, src2 YMMReg) {
	e.emitVexArith256Pp(opcode, dst, src1, src2, 0)
}

// emitVexArith256Pp is like emitVexArith256 but with an explicit pp field
// (0=none, 1=66, 2=F3, 3=F2). Used for instructions like vhaddps that
// require the F2 prefix in VEX.
func (e *Encoder) emitVexArith256Pp(opcode uint8, dst, src1, src2 YMMReg, pp uint8) {
	if src2 >= 8 {
		r := uint8(1)
		if dst >= 8 {
			r = 0
		}
		b := uint8(0)
		e.vex3(r, 1, b, 1, 0, uint8(src1), 1, pp)
	} else {
		r := uint8(1)
		if dst >= 8 {
			r = 0
		}
		e.vex2(r, uint8(src1), 1, pp)
	}
	e.Code = append(e.Code, opcode)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src2)))
}

// HAddPSReg128 emits `haddps dst, src` (F2 0F 7C /r), SSE3 horizontal add
// of packed f32. The 4 lanes of dst become:
//   dst[0] = dst[0] + dst[1]
//   dst[1] = dst[2] + dst[3]
//   dst[2] = src[0] + src[1]
//   dst[3] = src[2] + src[3]
// Two haddps in sequence reduce 4 lanes to a scalar in lane 0.
func (e *Encoder) HAddPSReg128(dst, src XMMReg) {
	e.Code = append(e.Code, 0xF2)
	rr := uint8(0)
	rb := uint8(0)
	if dst >= XMM8 {
		rr = 1
	}
	if src >= XMM8 {
		rb = 1
	}
	if rr|rb != 0 {
		e.Code = append(e.Code, 0x40|(rr<<2)|rb)
	}
	e.Code = append(e.Code, 0x0F, 0x7C)
	e.Code = append(e.Code, modrm(0b11, uint8(dst), uint8(src)))
}

// VHAddPS256 emits `vhaddps dst, src1, src2` (VEX.256.F2.0F.WIG 7C /r),
// horizontal add of 8 packed f32 in YMM. Lanes are interleaved like the
// 128-bit version, doubled.
func (e *Encoder) VHAddPS256(dst, src1, src2 YMMReg) {
	e.emitVexArith256Pp(0x7C, dst, src1, src2, 3) // pp=11=F2
}

// VExtractF128 emits `vextractf128 dst, src, imm8`
// (VEX.256.66.0F3A.W0 19 /r ib).
// imm8=0 extracts the low 128 bits, imm8=1 extracts the high 128 bits
// of the 256-bit src into the 128-bit dst (XMM register form).
func (e *Encoder) VExtractF128(dst XMMReg, src YMMReg, imm uint8) {
	// Always 3-byte VEX because of the 0F3A map.
	r := uint8(1)
	if src >= 8 {
		r = 0
	}
	b := uint8(1)
	if dst >= XMM8 {
		b = 0
	}
	// vex3(r, x, b, m, w, vvvv, L, pp): m=3 (0F3A), w=0, vvvv=0 (unused), L=1 (256-bit), pp=01 (66)
	e.vex3(r, 1, b, 3, 0, 0, 1, 1)
	e.Code = append(e.Code, 0x19)
	e.Code = append(e.Code, modrm(0b11, uint8(src), uint8(dst)))
	e.Code = append(e.Code, imm&1)
}
