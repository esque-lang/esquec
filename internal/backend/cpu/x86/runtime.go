package x86

import "encoding/binary"

// BuildPrintI32 synthesizes a CompiledFn for the runtime function
// `print_i32(x: i32) -> i32`. It writes the decimal representation of
// `x` followed by a newline to stdout (fd 1) via the write(2) syscall,
// then returns `x` unchanged so the call can be used in expressions
// (e.g. `let y = print_i32(x) * 2`).
//
// Calling convention: input in RDI (SysV), output in RAX.
//
// The function is callee-saved with respect to RBX, RBP, R12-R15 — it
// uses RBP for a frame pointer, restores it on exit, and otherwise
// touches only caller-saved scratch (RAX, RCX, RDX, RDI, RSI, R10).
//
// Algorithm: itoa into a 16-byte stack buffer, writing digits from the
// end. The last byte is '\n'. Digits are produced via signed division
// by 10. A leading '-' is prepended if the input was negative. Then a
// single write(1, buf+off, len) syscall flushes the formatted string.
func BuildPrintI32() *CompiledFn {
	enc := &Encoder{}

	// fwdJcc emits a 6-byte conditional jump with a placeholder 32-bit
	// displacement. Returns the offset of the displacement field so the
	// caller can patch it once the target position is known.
	fwdJcc := func(cc CondCode) int {
		enc.JccRel32(cc, 0)
		return len(enc.Code) - 4
	}
	fwdJmp := func() int {
		enc.JmpRel32(0)
		return len(enc.Code) - 4
	}
	patchRel := func(dispOff, target int) {
		instrEnd := dispOff + 4
		rel := int32(target - instrEnd)
		binary.LittleEndian.PutUint32(enc.Code[dispOff:dispOff+4], uint32(rel))
	}
	backJcc := func(cc CondCode, target int) {
		// Emit then patch immediately so the relative displacement is
		// computed from the end of this jump instruction.
		dispOff := fwdJcc(cc)
		patchRel(dispOff, target)
	}

	// Prologue: push rbp; mov rbp, rsp; sub rsp, 32
	// 32 bytes is more than enough for "-2147483648\n" (12 chars), keeps
	// the stack 16-byte aligned for the (immediate) syscall.
	enc.Push(RBP)
	enc.MovRegToReg64(RBP, RSP)
	enc.SubImm32FromReg64(RSP, 32)

	// r10 = rdi  (preserve original value for the return path)
	enc.MovRegToReg64(R10, RDI)

	// rax = abs(rdi):  rax = rdi; if rax < 0: neg rax
	enc.MovRegToReg64(RAX, RDI)
	enc.TestRegToReg64(RAX, RAX)
	jPos := fwdJcc(CCGE)
	enc.NegReg64(RAX)
	patchRel(jPos, len(enc.Code))

	// Last byte of the buffer (offset 15) is the newline.
	enc.MovByteImmToMem(RSP, 15, 0x0A)

	// rcx = &buf[14]  (next byte to write; we fill backwards from here)
	enc.LEA(RCX, RSP, 14)

	// If x == 0, we still need to emit a literal '0' — the loop body
	// would terminate before producing any digits.
	enc.TestRegToReg64(RAX, RAX)
	jNonZero := fwdJcc(CCNE)
	enc.MovByteImmToMem(RCX, 0, 0x30) // '0'
	enc.SubImm32FromReg64(RCX, 1)
	jToSign := fwdJmp()

	// .loop: while rax != 0 { rdx = rax % 10; rax /= 10; *rcx-- = '0'+rdx }
	loopStart := len(enc.Code)
	patchRel(jNonZero, loopStart)
	enc.MovImm32ToReg64(RDI, 10) // divisor
	enc.CQO()                    // sign-extend rax into rdx:rax
	enc.IDivReg64(RDI)           // rax = quot, rdx = rem
	enc.AddImm32ToReg64(RDX, 0x30)
	enc.MovByteRegToMem(RCX, 0, RDX) // store low byte of rdx
	enc.SubImm32FromReg64(RCX, 1)
	enc.TestRegToReg64(RAX, RAX)
	backJcc(CCNE, loopStart)

	// .signcheck: if r10 < 0, write a leading '-'
	signCheck := len(enc.Code)
	patchRel(jToSign, signCheck)
	enc.TestRegToReg64(R10, R10)
	jWrite := fwdJcc(CCGE)
	enc.MovByteImmToMem(RCX, 0, 0x2D) // '-'
	enc.SubImm32FromReg64(RCX, 1)
	patchRel(jWrite, len(enc.Code))

	// rcx currently points one byte before the first valid char.
	// Advance to the first char and compute length = (rsp+16) - rcx.
	enc.AddImm32ToReg64(RCX, 1)
	enc.LEA(RSI, RSP, 16)
	enc.SubRegFromReg64(RSI, RCX)
	enc.MovRegToReg64(RDX, RSI) // rdx = length
	enc.MovRegToReg64(RSI, RCX) // rsi = buffer ptr
	enc.MovImm32ToReg64(RDI, 1) // fd = stdout
	enc.MovImm32ToReg64(RAX, 1) // sys_write
	enc.SyscallInsn()

	// Return original input in RAX.
	enc.MovRegToReg64(RAX, R10)

	// Epilogue
	enc.AddImm32ToReg64(RSP, 32)
	enc.Pop(RBP)
	enc.Ret()

	return &CompiledFn{
		Name: "print_i32",
		Code: enc.Code,
		Rels: nil,
	}
}

// BuildPrintF32 synthesizes a CompiledFn for the runtime function
// `print_f32(x: f32) -> f32`. It writes the decimal representation of
// `x` with six fractional digits followed by a newline to stdout, then
// returns `x` unchanged so the call can be used in expressions.
//
// Calling convention: input in XMM0, output in XMM0.
//
// Algorithm: split the absolute value into an integer part (truncated
// toward zero with cvttss2si) and a fractional part scaled by 1e6 and
// truncated to an integer in the range 0..999999. The integer part is
// itoa'd backwards into a stack buffer like print_i32, the six
// fractional digits are emitted with leading zeros, the two are
// joined by '.', and a leading '-' is prepended for negative inputs.
//
// Limitations (deferred): NaN, ±Inf, and values whose magnitude
// exceeds 2^31 produce undefined output (cvttss2si returns
// 0x80000000 for these and the formatter does not special-case it).
// Rounding is truncation, not half-to-even — print_f32(0.9999996) is
// "0.999999" rather than "1.000000".
func BuildPrintF32() *CompiledFn {
	enc := &Encoder{}

	fwdJcc := func(cc CondCode) int {
		enc.JccRel32(cc, 0)
		return len(enc.Code) - 4
	}
	fwdJmp := func() int {
		enc.JmpRel32(0)
		return len(enc.Code) - 4
	}
	patchRel := func(dispOff, target int) {
		instrEnd := dispOff + 4
		rel := int32(target - instrEnd)
		binary.LittleEndian.PutUint32(enc.Code[dispOff:dispOff+4], uint32(rel))
	}
	backJcc := func(cc CondCode, target int) {
		dispOff := fwdJcc(cc)
		patchRel(dispOff, target)
	}

	// Stack layout (48 bytes; rsp 16-aligned at syscall):
	//   [rsp+0..31]   formatted output buffer (right-aligned, '\n' at +31)
	//   [rsp+32..35]  saved integer part of |x|
	//   [rsp+40..43]  saved original f32 (returned in XMM0)
	//   [rsp+44..47]  scratch slot for 1e6 constant
	enc.Push(RBP)
	enc.MovRegToReg64(RBP, RSP)
	enc.SubImm32FromReg64(RSP, 48)

	// Save original input for the return value.
	enc.MovSSStoreMem(RSP, 40, XMM0)

	// Pull the bit pattern into r10d so we can sign-test without leaving
	// the integer pipe. The 32-bit load zero-extends, so we must use
	// the 32-bit form of `test` to expose the original sign bit.
	enc.MovLoadMem32(R10, RSP, 40)

	// r11 holds the sign indicator: 0 for non-negative, 1 for negative.
	enc.MovImm32ToReg64(R11, 0)
	enc.TestRegToReg32(R10, R10)
	jNonNeg := fwdJcc(CCGE)

	// Negative branch: record the sign and replace XMM0 with -XMM0
	// (computed as 0 - XMM0 in XMM1, then full-128-bit copied back).
	enc.MovImm32ToReg64(R11, 1)
	enc.XorPSRegToReg(XMM1, XMM1)
	enc.SubSSRegToReg(XMM1, XMM0)
	enc.MovUPSReg128(XMM0, XMM1)

	patchRel(jNonNeg, len(enc.Code))

	// Integer part: rax = (int)|x|, then stash so the digit loops don't
	// have to keep it in a register across cvtsi2ss / mulss / cvttss2si.
	enc.CVTTSS2SI(RAX, XMM0)
	enc.MovStoreMem32(RSP, 32, RAX)

	// Fractional part: xmm0 -= (float)rax, then *= 1e6, then truncate
	// to rcx (0..999999). 1e6 as f32 is 0x49742400.
	enc.CVTSI2SS(XMM1, RAX)
	enc.SubSSRegToReg(XMM0, XMM1)
	enc.MovImm32ToMem(RSP, 44, 0x49742400)
	enc.MovSSLoadMem(XMM1, RSP, 44)
	enc.MulSSRegToReg(XMM0, XMM1)
	enc.CVTTSS2SI(RCX, XMM0)

	// Buffer formatting (right-to-left from [rsp+31]):
	// [rsp+31] = '\n'; rdi = &[rsp+30] is the next byte to write.
	enc.MovByteImmToMem(RSP, 31, 0x0A)
	enc.LEA(RDI, RSP, 30)

	// Six fractional digits with leading zeros. We feed rax with the
	// frac value once and rely on idiv leaving the next quotient in
	// rax for the following iteration. r8 holds the divisor.
	enc.MovRegToReg64(RAX, RCX)
	enc.MovImm32ToReg64(R8, 10)
	enc.MovImm32ToReg64(RSI, 6)
	fracLoop := len(enc.Code)
	enc.CQO()
	enc.IDivReg64(R8)
	enc.AddImm32ToReg64(RDX, 0x30)
	enc.MovByteRegToMem(RDI, 0, RDX)
	enc.SubImm32FromReg64(RDI, 1)
	enc.SubImm32FromReg64(RSI, 1)
	backJcc(CCNE, fracLoop)

	// Decimal point.
	enc.MovByteImmToMem(RDI, 0, 0x2E)
	enc.SubImm32FromReg64(RDI, 1)

	// Integer part: reload rax from the stash. If the integer part is
	// zero we still need to emit a literal '0' (the digit loop would
	// produce nothing).
	enc.MovLoadMem32(RAX, RSP, 32)
	enc.TestRegToReg64(RAX, RAX)
	jIntNonZero := fwdJcc(CCNE)
	enc.MovByteImmToMem(RDI, 0, 0x30)
	enc.SubImm32FromReg64(RDI, 1)
	jToSign := fwdJmp()

	patchRel(jIntNonZero, len(enc.Code))
	intLoop := len(enc.Code)
	enc.CQO()
	enc.IDivReg64(R8)
	enc.AddImm32ToReg64(RDX, 0x30)
	enc.MovByteRegToMem(RDI, 0, RDX)
	enc.SubImm32FromReg64(RDI, 1)
	enc.TestRegToReg64(RAX, RAX)
	backJcc(CCNE, intLoop)

	patchRel(jToSign, len(enc.Code))

	// Optional leading '-'.
	enc.TestRegToReg64(R11, R11)
	jNoSign := fwdJcc(CCE)
	enc.MovByteImmToMem(RDI, 0, 0x2D)
	enc.SubImm32FromReg64(RDI, 1)
	patchRel(jNoSign, len(enc.Code))

	// rdi is one byte before the first valid char. Advance, then
	// length = (rsp+32) - rdi. Set up sys_write(1, ptr, len).
	enc.AddImm32ToReg64(RDI, 1)
	enc.LEA(RSI, RSP, 32)
	enc.SubRegFromReg64(RSI, RDI)
	enc.MovRegToReg64(RDX, RSI) // rdx = length
	enc.MovRegToReg64(RSI, RDI) // rsi = buffer
	enc.MovImm32ToReg64(RDI, 1) // fd = stdout
	enc.MovImm32ToReg64(RAX, 1) // sys_write
	enc.SyscallInsn()

	// Restore original f32 to xmm0 for return.
	enc.MovSSLoadMem(XMM0, RSP, 40)

	// Epilogue.
	enc.AddImm32ToReg64(RSP, 48)
	enc.Pop(RBP)
	enc.Ret()

	return &CompiledFn{
		Name: "print_f32",
		Code: enc.Code,
		Rels: nil,
	}
}

// BuildPrintStr synthesizes a CompiledFn for the runtime function
// `print_str(s: string) -> string`. The fat-pointer string ABI passes
// the buffer address in RDI and the length in RSI (System V argument
// slots 0 and 1). This function calls write(2) — `write(1, ptr, len)`
// — and returns the same fat pointer in (RAX, RDX) so callers can
// chain. v0.12 has no codepath that consumes the return value, but
// emitting it keeps the ABI symmetric with print_i32 / print_f32.
//
// The syscall convention is:
//   RAX = 1 (sys_write)
//   RDI = 1 (fd = stdout)
//   RSI = buffer pointer
//   RDX = length
//
// We arrive with RDI = ptr, RSI = len, so we save the pointer and
// length into callee-preserved scratch (R10, R11) before reshuffling
// the syscall argument registers, then restore them into the
// return-value slots after the syscall completes.
func BuildPrintStr() *CompiledFn {
	enc := &Encoder{}

	// r10 = rdi (saved ptr)   r11 = rsi (saved len)
	enc.MovRegToReg64(R10, RDI)
	enc.MovRegToReg64(R11, RSI)

	// rdx = len   rsi = ptr   rdi = 1   rax = 1
	enc.MovRegToReg64(RDX, R11)
	enc.MovRegToReg64(RSI, R10)
	enc.MovImm32ToReg64(RDI, 1)
	enc.MovImm32ToReg64(RAX, 1)
	enc.SyscallInsn()

	// Return (ptr, len) in (RAX, RDX) per the fat-pointer ABI.
	enc.MovRegToReg64(RAX, R10)
	enc.MovRegToReg64(RDX, R11)
	enc.Ret()

	return &CompiledFn{
		Name: "print_str",
		Code: enc.Code,
		Rels: nil,
	}
}

// IsPrintBuiltin reports whether `name` is a runtime print intrinsic
// that the backend will synthesize on demand. Other layers (type
// checker, CEIR lowering, link driver) consult this so they don't try
// to emit a body for these names from user-space.
func IsPrintBuiltin(name string) bool {
	switch name {
	case "print_i32", "print_f32", "print_str":
		return true
	}
	return false
}
