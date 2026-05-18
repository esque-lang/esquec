// Package x86 is the CPU code generator for x86-64 SysV (Linux).
package x86

// Reg is a general-purpose 64-bit register encoding.
type Reg uint8

const (
	RAX Reg = 0
	RCX Reg = 1
	RDX Reg = 2
	RBX Reg = 3
	RSP Reg = 4
	RBP Reg = 5
	RSI Reg = 6
	RDI Reg = 7
	R8  Reg = 8
	R9  Reg = 9
	R10 Reg = 10
	R11 Reg = 11
	R12 Reg = 12
	R13 Reg = 13
	R14 Reg = 14
	R15 Reg = 15
)

// SysV AMD64 integer argument registers.
var SysVIntArgs = []Reg{RDI, RSI, RDX, RCX, R8, R9}

// SysVCallerSaved are the registers a function must assume are clobbered
// across a call.
var SysVCallerSaved = []Reg{RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11}

// SysVCalleeSaved must be preserved across a call.
var SysVCalleeSaved = []Reg{RBX, RBP, R12, R13, R14, R15}

// XMMReg is an SSE/AVX 128-bit register encoding.
type XMMReg uint8

const (
	XMM0  XMMReg = 0
	XMM1  XMMReg = 1
	XMM2  XMMReg = 2
	XMM3  XMMReg = 3
	XMM4  XMMReg = 4
	XMM5  XMMReg = 5
	XMM6  XMMReg = 6
	XMM7  XMMReg = 7
	XMM8  XMMReg = 8
	XMM9  XMMReg = 9
	XMM10 XMMReg = 10
	XMM11 XMMReg = 11
	XMM12 XMMReg = 12
	XMM13 XMMReg = 13
	XMM14 XMMReg = 14
	XMM15 XMMReg = 15
)

// SysVFloatArgs are the XMM registers used for float arguments in SysV ABI.
var SysVFloatArgs = []XMMReg{XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7}

// SysVFloatCallerSaved are XMM registers that are caller-saved.
// All XMM registers are caller-saved in SysV ABI.
var SysVFloatCallerSaved = []XMMReg{
	XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
	XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
}

// YMMReg is an AVX 256-bit register encoding.
// YMM0-15 share lower 128 bits with XMM0-15.
type YMMReg uint8

const (
	YMM0  YMMReg = 0
	YMM1  YMMReg = 1
	YMM2  YMMReg = 2
	YMM3  YMMReg = 3
	YMM4  YMMReg = 4
	YMM5  YMMReg = 5
	YMM6  YMMReg = 6
	YMM7  YMMReg = 7
	YMM8  YMMReg = 8
	YMM9  YMMReg = 9
	YMM10 YMMReg = 10
	YMM11 YMMReg = 11
	YMM12 YMMReg = 12
	YMM13 YMMReg = 13
	YMM14 YMMReg = 14
	YMM15 YMMReg = 15
)
