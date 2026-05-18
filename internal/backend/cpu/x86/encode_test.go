package x86

import (
	"strings"
	"testing"

	"golang.org/x/arch/x86/x86asm"
)

// decode helper: decode a single x86-64 instruction from b and return
// it in Intel syntax.
func decode(t *testing.T, b []byte) string {
	t.Helper()
	insn, err := x86asm.Decode(b, 64)
	if err != nil {
		t.Fatalf("x86asm.Decode(%x): %v", b, err)
	}
	if insn.Len != len(b) {
		t.Fatalf("decode consumed %d of %d bytes for %x", insn.Len, len(b), b)
	}
	return strings.ToLower(x86asm.IntelSyntax(insn, 0, nil))
}

func TestMovImm32ToReg64(t *testing.T) {
	cases := []struct {
		reg    Reg
		imm    int32
		expect string
	}{
		{RAX, 42, "mov rax, 0x2a"},
		{RDI, 60, "mov rdi, 0x3c"},
		{R10, 1, "mov r10, 0x1"},
	}
	for _, c := range cases {
		var e Encoder
		e.MovImm32ToReg64(c.reg, c.imm)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("MovImm32ToReg64(%v, %d) -> %q, want %q (bytes=%x)",
				c.reg, c.imm, got, c.expect, e.Code)
		}
	}
}

func TestMovRegToReg64(t *testing.T) {
	cases := []struct {
		dst, src Reg
		expect   string
	}{
		{RAX, RDI, "mov rax, rdi"},
		{RDI, RAX, "mov rdi, rax"},
		{R10, RAX, "mov r10, rax"},
	}
	for _, c := range cases {
		var e Encoder
		e.MovRegToReg64(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("MovRegToReg64(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestAddRegToReg64(t *testing.T) {
	var e Encoder
	e.AddRegToReg64(RAX, RDI)
	got := decode(t, e.Code)
	want := "add rax, rdi"
	if got != want {
		t.Errorf("AddRegToReg64 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestSubRegFromReg64(t *testing.T) {
	var e Encoder
	e.SubRegFromReg64(RAX, RDI)
	got := decode(t, e.Code)
	want := "sub rax, rdi"
	if got != want {
		t.Errorf("SubRegFromReg64 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestIMulRegByReg64(t *testing.T) {
	var e Encoder
	e.IMulRegByReg64(RAX, RDI)
	got := decode(t, e.Code)
	want := "imul rax, rdi"
	if got != want {
		t.Errorf("IMulRegByReg64 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestNegReg64(t *testing.T) {
	var e Encoder
	e.NegReg64(RAX)
	got := decode(t, e.Code)
	want := "neg rax"
	if got != want {
		t.Errorf("NegReg64 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestRet(t *testing.T) {
	var e Encoder
	e.Ret()
	if got, want := decode(t, e.Code), "ret"; got != want {
		t.Errorf("Ret -> %q, want %q", got, want)
	}
}

func TestSyscall(t *testing.T) {
	var e Encoder
	e.SyscallInsn()
	if got, want := decode(t, e.Code), "syscall"; got != want {
		t.Errorf("Syscall -> %q, want %q", got, want)
	}
}

// ============================================================
// SSE Instruction Tests
// ============================================================

func TestMovSSRegToReg(t *testing.T) {
	cases := []struct {
		dst, src XMMReg
		expect   string
	}{
		{XMM0, XMM1, "movss xmm0, xmm1"},
		{XMM1, XMM0, "movss xmm1, xmm0"},
		{XMM8, XMM0, "movss xmm8, xmm0"},
		{XMM0, XMM8, "movss xmm0, xmm8"},
		{XMM10, XMM12, "movss xmm10, xmm12"},
	}
	for _, c := range cases {
		var e Encoder
		e.MovSSRegToReg(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("MovSSRegToReg(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestAddSSRegToReg(t *testing.T) {
	cases := []struct {
		dst, src XMMReg
		expect   string
	}{
		{XMM0, XMM1, "addss xmm0, xmm1"},
		{XMM5, XMM3, "addss xmm5, xmm3"},
		{XMM8, XMM9, "addss xmm8, xmm9"},
	}
	for _, c := range cases {
		var e Encoder
		e.AddSSRegToReg(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("AddSSRegToReg(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestSubSSRegToReg(t *testing.T) {
	var e Encoder
	e.SubSSRegToReg(XMM0, XMM1)
	got := decode(t, e.Code)
	want := "subss xmm0, xmm1"
	if got != want {
		t.Errorf("SubSSRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMulSSRegToReg(t *testing.T) {
	var e Encoder
	e.MulSSRegToReg(XMM2, XMM3)
	got := decode(t, e.Code)
	want := "mulss xmm2, xmm3"
	if got != want {
		t.Errorf("MulSSRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestDivSSRegToReg(t *testing.T) {
	var e Encoder
	e.DivSSRegToReg(XMM4, XMM5)
	got := decode(t, e.Code)
	want := "divss xmm4, xmm5"
	if got != want {
		t.Errorf("DivSSRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

// ============================================================
// SSE2 f64 Scalar Instruction Tests
// ============================================================

func TestMovSDRegToReg(t *testing.T) {
	cases := []struct {
		dst, src XMMReg
		expect   string
	}{
		{XMM0, XMM1, "movsd xmm0, xmm1"},
		{XMM8, XMM0, "movsd xmm8, xmm0"},
		{XMM10, XMM12, "movsd xmm10, xmm12"},
	}
	for _, c := range cases {
		var e Encoder
		e.MovSDRegToReg(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("MovSDRegToReg(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestAddSDRegToReg(t *testing.T) {
	var e Encoder
	e.AddSDRegToReg(XMM0, XMM1)
	got := decode(t, e.Code)
	want := "addsd xmm0, xmm1"
	if got != want {
		t.Errorf("AddSDRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestSubSDRegToReg(t *testing.T) {
	var e Encoder
	e.SubSDRegToReg(XMM2, XMM3)
	got := decode(t, e.Code)
	want := "subsd xmm2, xmm3"
	if got != want {
		t.Errorf("SubSDRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMulSDRegToReg(t *testing.T) {
	var e Encoder
	e.MulSDRegToReg(XMM4, XMM5)
	got := decode(t, e.Code)
	want := "mulsd xmm4, xmm5"
	if got != want {
		t.Errorf("MulSDRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestDivSDRegToReg(t *testing.T) {
	var e Encoder
	e.DivSDRegToReg(XMM6, XMM7)
	got := decode(t, e.Code)
	want := "divsd xmm6, xmm7"
	if got != want {
		t.Errorf("DivSDRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMovSDLoadMem(t *testing.T) {
	var e Encoder
	e.MovSDLoadMem(XMM0, RSP, -8)
	got := decode(t, e.Code)
	// x86asm decodes a 32-bit displacement as unsigned, so -8 is shown
	// as 0xfffffff8.
	want := "movsd xmm0, qword ptr [rsp+0xfffffff8]"
	if got != want {
		t.Errorf("MovSDLoadMem -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMovSDStoreMem(t *testing.T) {
	var e Encoder
	e.MovSDStoreMem(RSP, -8, XMM1)
	got := decode(t, e.Code)
	want := "movsd qword ptr [rsp+0xfffffff8], xmm1"
	if got != want {
		t.Errorf("MovSDStoreMem -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestXorPSRegToReg(t *testing.T) {
	var e Encoder
	e.XorPSRegToReg(XMM0, XMM0)
	got := decode(t, e.Code)
	want := "xorps xmm0, xmm0"
	if got != want {
		t.Errorf("XorPSRegToReg -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestJmpRel32(t *testing.T) {
	var e Encoder
	e.JmpRel32(10)
	got := decode(t, e.Code)
	want := "jmp .+0xa" // jmp rel32, displacement is 10
	if got != want {
		t.Errorf("JmpRel32 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestLEA(t *testing.T) {
	var e Encoder
	e.LEA(RAX, RSP, 16)
	got := decode(t, e.Code)
	want := "lea rax, ptr [rsp+0x10]" // x86asm includes "ptr"
	if got != want {
		t.Errorf("LEA -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

// ============================================================
// Conversion Instruction Tests
// ============================================================

func TestCVTTSS2SI(t *testing.T) {
	cases := []struct {
		dst    Reg
		src    XMMReg
		expect string
	}{
		{RAX, XMM0, "cvttss2si eax, xmm0"},
		{RCX, XMM1, "cvttss2si ecx, xmm1"},
		{R10, XMM8, "cvttss2si r10d, xmm8"},
	}
	for _, c := range cases {
		var e Encoder
		e.CVTTSS2SI(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("CVTTSS2SI(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestCVTSI2SS(t *testing.T) {
	cases := []struct {
		dst    XMMReg
		src    Reg
		expect string
	}{
		{XMM0, RAX, "cvtsi2ss xmm0, eax"},
		{XMM1, RCX, "cvtsi2ss xmm1, ecx"},
		{XMM8, R10, "cvtsi2ss xmm8, r10d"},
	}
	for _, c := range cases {
		var e Encoder
		e.CVTSI2SS(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("CVTSI2SS(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestCVTTSD2SI(t *testing.T) {
	cases := []struct {
		dst    Reg
		src    XMMReg
		expect string
	}{
		{RAX, XMM0, "cvttsd2si eax, xmm0"},
		{RCX, XMM1, "cvttsd2si ecx, xmm1"},
		{R10, XMM8, "cvttsd2si r10d, xmm8"},
	}
	for _, c := range cases {
		var e Encoder
		e.CVTTSD2SI(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("CVTTSD2SI(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestCVTSI2SD(t *testing.T) {
	cases := []struct {
		dst    XMMReg
		src    Reg
		expect string
	}{
		{XMM0, RAX, "cvtsi2sd xmm0, eax"},
		{XMM1, RCX, "cvtsi2sd xmm1, ecx"},
		{XMM8, R10, "cvtsi2sd xmm8, r10d"},
	}
	for _, c := range cases {
		var e Encoder
		e.CVTSI2SD(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("CVTSI2SD(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestMovImm64ToMem(t *testing.T) {
	var e Encoder
	e.MovImm64ToMem(RSP, -8, 0x1122334455667788)
	// Two `mov dword [rsp+disp32], imm32` stores: low half at -8, high
	// half at -4. Each [rsp+disp32] addressing form needs a SIB byte,
	// so each store is C7 84 24 + 4-byte disp + 4-byte imm = 11 bytes.
	if len(e.Code) != 22 {
		t.Fatalf("MovImm64ToMem produced %d bytes, want 22 (bytes=%x)",
			len(e.Code), e.Code)
	}
	got := decode(t, e.Code[:11])
	if got != "mov dword ptr [rsp+0xfffffff8], 0x55667788" {
		t.Errorf("MovImm64ToMem low -> %q (bytes=%x)", got, e.Code[:11])
	}
	got2 := decode(t, e.Code[11:])
	if got2 != "mov dword ptr [rsp+0xfffffffc], 0x11223344" {
		t.Errorf("MovImm64ToMem high -> %q (bytes=%x)", got2, e.Code[11:])
	}
}

// ============================================================
// AVX2 Instruction Tests
// Note: x86asm library doesn't support VEX prefix decoding,
// so we verify bytes directly against known-good encodings.
// Encodings verified with objdump.
// ============================================================

func TestVAddPS256(t *testing.T) {
	cases := []struct {
		dst, src1, src2 YMMReg
		expectBytes     string // hex bytes
	}{
		// VADDPS YMM0, YMM1, YMM2: C5 F4 58 C2
		// VEX.256.0F 58 /r
		// C5 = 2-byte VEX, F4 = R=1, vvvv=~1=E, L=1, pp=0
		// 58 = VADDPS opcode, C2 = mod=11, reg=0, rm=2
		{YMM0, YMM1, YMM2, "c5 f4 58 c2"},
		// VADDPS YMM1, YMM2, YMM3: C5 EC 58 CB
		{YMM1, YMM2, YMM3, "c5 ec 58 cb"},
		// VADDPS YMM7, YMM0, YMM1: C5 FC 58 F9
		{YMM7, YMM0, YMM1, "c5 fc 58 f9"},
	}
	for _, c := range cases {
		var e Encoder
		e.VAddPS256(c.dst, c.src1, c.src2)
		got := e.Hex()
		if got != c.expectBytes {
			t.Errorf("VAddPS256(%v, %v, %v) -> %q, want %q",
				c.dst, c.src1, c.src2, got, c.expectBytes)
		}
	}
}

func TestVSubPS256(t *testing.T) {
	var e Encoder
	e.VSubPS256(YMM0, YMM1, YMM2)
	got := e.Hex()
	// VSUBPS YMM0, YMM1, YMM2: C5 F4 5C C2
	want := "c5 f4 5c c2"
	if got != want {
		t.Errorf("VSubPS256 -> %q, want %q", got, want)
	}
}

func TestVMulPS256(t *testing.T) {
	var e Encoder
	e.VMulPS256(YMM3, YMM4, YMM5)
	got := e.Hex()
	// VMULPS YMM3, YMM4, YMM5: C5 DC 59 DD
	want := "c5 dc 59 dd"
	if got != want {
		t.Errorf("VMulPS256 -> %q, want %q", got, want)
	}
}

func TestVDivPS256(t *testing.T) {
	var e Encoder
	e.VDivPS256(YMM6, YMM7, YMM0)
	got := e.Hex()
	// VDIVPS YMM6, YMM7, YMM0: C5 C4 5E F0
	want := "c5 c4 5e f0"
	if got != want {
		t.Errorf("VDivPS256 -> %q, want %q", got, want)
	}
}

func TestVMovUPSLoad256(t *testing.T) {
	var e Encoder
	e.VMovUPSLoad256(YMM0, RAX, 32)
	got := e.Hex()
	// VMOVUPS YMM0, [RAX+32]: C5 FC 10 80 20000000
	want := "c5 fc 10 80 20 00 00 00"
	if got != want {
		t.Errorf("VMovUPSLoad256 -> %q, want %q", got, want)
	}
}

func TestVMovUPSStore256(t *testing.T) {
	var e Encoder
	e.VMovUPSStore256(RAX, 64, YMM1)
	got := e.Hex()
	// VMOVUPS [RAX+64], YMM1: C5 FC 11 88 40000000
	want := "c5 fc 11 88 40 00 00 00"
	if got != want {
		t.Errorf("VMovUPSStore256 -> %q, want %q", got, want)
	}
}

// ============================================================
// SSE Packed Instruction Tests (128-bit, 4 x f32)
// ============================================================

func TestMovUPSLoad128(t *testing.T) {
	var e Encoder
	e.MovUPSLoad128(XMM0, RAX, 16)
	got := decode(t, e.Code)
	want := "movups xmm0, xmmword ptr [rax+0x10]"
	if got != want {
		t.Errorf("MovUPSLoad128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMovUPSStore128(t *testing.T) {
	var e Encoder
	e.MovUPSStore128(RAX, 32, XMM1)
	got := decode(t, e.Code)
	want := "movups xmmword ptr [rax+0x20], xmm1"
	if got != want {
		t.Errorf("MovUPSStore128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMovUPSReg128(t *testing.T) {
	var e Encoder
	e.MovUPSReg128(XMM0, XMM1)
	got := decode(t, e.Code)
	want := "movups xmm0, xmm1"
	if got != want {
		t.Errorf("MovUPSReg128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestAddPSReg128(t *testing.T) {
	var e Encoder
	e.AddPSReg128(XMM0, XMM1)
	got := decode(t, e.Code)
	want := "addps xmm0, xmm1"
	if got != want {
		t.Errorf("AddPSReg128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestSubPSReg128(t *testing.T) {
	var e Encoder
	e.SubPSReg128(XMM2, XMM3)
	got := decode(t, e.Code)
	want := "subps xmm2, xmm3"
	if got != want {
		t.Errorf("SubPSReg128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestMulPSReg128(t *testing.T) {
	var e Encoder
	e.MulPSReg128(XMM4, XMM5)
	got := decode(t, e.Code)
	want := "mulps xmm4, xmm5"
	if got != want {
		t.Errorf("MulPSReg128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

func TestDivPSReg128(t *testing.T) {
	var e Encoder
	e.DivPSReg128(XMM6, XMM7)
	got := decode(t, e.Code)
	want := "divps xmm6, xmm7"
	if got != want {
		t.Errorf("DivPSReg128 -> %q, want %q (bytes=%x)", got, want, e.Code)
	}
}

// ============================================================
// Horizontal SIMD reduction primitives
// ============================================================

func TestHAddPSReg128(t *testing.T) {
	cases := []struct {
		dst, src XMMReg
		expect   string
	}{
		{XMM0, XMM1, "haddps xmm0, xmm1"},
		{XMM2, XMM3, "haddps xmm2, xmm3"},
		{XMM7, XMM0, "haddps xmm7, xmm0"},
	}
	for _, c := range cases {
		var e Encoder
		e.HAddPSReg128(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("HAddPSReg128(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestHAddPSReg128Bytes(t *testing.T) {
	// HADDPS xmm0, xmm1: F2 0F 7C C1
	var e Encoder
	e.HAddPSReg128(XMM0, XMM1)
	got := e.Hex()
	want := "f2 0f 7c c1"
	if got != want {
		t.Errorf("HAddPSReg128(XMM0, XMM1) bytes = %q, want %q", got, want)
	}
}

func TestVHAddPS256(t *testing.T) {
	cases := []struct {
		dst, src1, src2 YMMReg
		expect          string // hex bytes
	}{
		// VHADDPS YMM0, YMM1, YMM2: C5 F7 7C C2
		// 2-byte VEX: R=1, vvvv=~1=0b1110, L=1, pp=11(F2) -> 0xF7
		{YMM0, YMM1, YMM2, "c5 f7 7c c2"},
		// VHADDPS YMM3, YMM4, YMM5: C5 DF 7C DD
		// vvvv=~4=0b1011 -> byte2 = 0x80 | 0xB<<3 | 0x4 | 0x3 = 0xDF
		{YMM3, YMM4, YMM5, "c5 df 7c dd"},
	}
	for _, c := range cases {
		var e Encoder
		e.VHAddPS256(c.dst, c.src1, c.src2)
		got := e.Hex()
		if got != c.expect {
			t.Errorf("VHAddPS256(%v, %v, %v) -> %q, want %q",
				c.dst, c.src1, c.src2, got, c.expect)
		}
	}
}

// TestDivReg64 (v0.11) verifies that DIV r/m64 emits with REX.W and
// /6 — i.e., that the unsigned-divide form is distinct from the signed
// IDIV form (/7) the encoder already had.
func TestDivReg64(t *testing.T) {
	cases := []struct {
		src    Reg
		expect string
	}{
		{RCX, "div rcx"},
		{R10, "div r10"},
	}
	for _, c := range cases {
		var e Encoder
		e.DivReg64(c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("DivReg64(%v) -> %q, want %q (bytes=%x)",
				c.src, got, c.expect, e.Code)
		}
	}
}

// TestSetCCUnsigned (v0.11) verifies the new unsigned condition codes
// (CCB, CCBE, CCA, CCAE) round-trip through SetCC into the expected
// SETcc opcodes (0F 92 / 96 / 97 / 93). The disassembler in use prints
// SETA as SETNBE and SETAE as SETNB — both are valid Intel synonyms
// for the same opcode byte; we match either form.
func TestSetCCUnsigned(t *testing.T) {
	cases := []struct {
		cc     CondCode
		expect []string // any of these is acceptable
	}{
		{CCB, []string{"setb al", "setnae al"}},
		{CCBE, []string{"setbe al", "setna al"}},
		{CCA, []string{"seta al", "setnbe al"}},
		{CCAE, []string{"setae al", "setnb al"}},
	}
	for _, c := range cases {
		var e Encoder
		e.SetCC(c.cc, RAX)
		got := decode(t, e.Code)
		ok := false
		for _, want := range c.expect {
			if got == want {
				ok = true
				break
			}
		}
		if !ok {
			t.Errorf("SetCC(%v, RAX) -> %q, want one of %v (bytes=%x)",
				c.cc, got, c.expect, e.Code)
		}
	}
}

func TestVExtractF128(t *testing.T) {
	cases := []struct {
		dst    XMMReg
		src    YMMReg
		imm    uint8
		expect string // hex bytes
	}{
		// VEXTRACTF128 XMM0, YMM1, 0: C4 E3 7D 19 C8 00
		// 3-byte VEX: byte2 = R=1 X=1 B=1 m=3 -> 0xE3
		// byte3: W=0 vvvv=~0=0xF L=1 pp=01(66) -> 0x7D
		// ModR/M: mod=11 reg=src=1 rm=dst=0 -> 0xC8
		{XMM0, YMM1, 0, "c4 e3 7d 19 c8 00"},
		// VEXTRACTF128 XMM2, YMM3, 1: C4 E3 7D 19 DA 01
		// ModR/M: reg=3 rm=2 -> 0xDA
		{XMM2, YMM3, 1, "c4 e3 7d 19 da 01"},
	}
	for _, c := range cases {
		var e Encoder
		e.VExtractF128(c.dst, c.src, c.imm)
		got := e.Hex()
		if got != c.expect {
			t.Errorf("VExtractF128(%v, %v, %d) -> %q, want %q",
				c.dst, c.src, c.imm, got, c.expect)
		}
	}
}

func TestMovzxReg8ToReg64(t *testing.T) {
	cases := []struct {
		dst, src Reg
		expect   string
	}{
		{RAX, RDI, "movzx rax, dil"},
		{RDX, RSI, "movzx rdx, sil"},
		{R10, RAX, "movzx r10, al"},
	}
	for _, c := range cases {
		var e Encoder
		e.MovzxReg8ToReg64(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("MovzxReg8ToReg64(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}

func TestMovsxReg8ToReg64(t *testing.T) {
	cases := []struct {
		dst, src Reg
		expect   string
	}{
		{RAX, RDI, "movsx rax, dil"},
		{RDX, RSI, "movsx rdx, sil"},
		{R10, RAX, "movsx r10, al"},
	}
	for _, c := range cases {
		var e Encoder
		e.MovsxReg8ToReg64(c.dst, c.src)
		got := decode(t, e.Code)
		if got != c.expect {
			t.Errorf("MovsxReg8ToReg64(%v, %v) -> %q, want %q (bytes=%x)",
				c.dst, c.src, got, c.expect, e.Code)
		}
	}
}
