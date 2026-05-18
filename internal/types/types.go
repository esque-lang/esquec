// Package types implements the esque type system.
package types

import (
	"fmt"
	"strings"
)

// Kind enumerates element type kinds.
type Kind int

const (
	KInvalid Kind = iota
	KBool
	KI8
	KI16
	KI32
	KI64
	KU8
	KU16
	KU32
	KU64
	KF32
	KF64
	KUnit
	KString  // Immutable UTF-8 string, fat pointer (ptr, len) (v0.12)
	KVec8F32 // Internal: 8 x f32 packed in YMM (not user-visible)
	KVec4F32 // Internal: 4 x f32 packed in XMM (not user-visible)
)

// ShapeDim represents a tensor dimension.
// Either a concrete value (Val >= 0) or a variable (Var != "").
type ShapeDim struct {
	Val int64  // concrete dimension value (-1 if variable)
	Var string // variable name (empty if concrete)
}

func (d ShapeDim) String() string {
	if d.Var != "" {
		return d.Var
	}
	return fmt.Sprintf("%d", d.Val)
}

func (d ShapeDim) Equal(e ShapeDim) bool {
	if d.Var != "" || e.Var != "" {
		return d.Var == e.Var
	}
	return d.Val == e.Val
}

// ConcreteDim creates a concrete dimension
func ConcreteDim(v int64) ShapeDim {
	return ShapeDim{Val: v, Var: ""}
}

// VarDim creates a variable dimension
func VarDim(name string) ShapeDim {
	return ShapeDim{Val: -1, Var: name}
}

// Type represents a value type.
// For scalars, Shape is nil. For tensors, Shape contains dimensions.
type Type struct {
	K     Kind
	Shape []ShapeDim // nil for scalars, non-nil for tensors
}

func Bool() *Type    { return &Type{K: KBool} }
func I8() *Type      { return &Type{K: KI8} }
func I16() *Type     { return &Type{K: KI16} }
func I32() *Type     { return &Type{K: KI32} }
func I64() *Type     { return &Type{K: KI64} }
func U8() *Type      { return &Type{K: KU8} }
func U16() *Type     { return &Type{K: KU16} }
func U32() *Type     { return &Type{K: KU32} }
func U64() *Type     { return &Type{K: KU64} }
func F32() *Type     { return &Type{K: KF32} }
func F64() *Type     { return &Type{K: KF64} }
func Unit() *Type    { return &Type{K: KUnit} }
func String() *Type  { return &Type{K: KString} }
func Vec8F32() *Type { return &Type{K: KVec8F32} }
func Vec4F32() *Type { return &Type{K: KVec4F32} }

func (t *Type) String() string {
	var base string
	switch t.K {
	case KBool:
		base = "bool"
	case KI8:
		base = "i8"
	case KI16:
		base = "i16"
	case KI32:
		base = "i32"
	case KI64:
		base = "i64"
	case KU8:
		base = "u8"
	case KU16:
		base = "u16"
	case KU32:
		base = "u32"
	case KU64:
		base = "u64"
	case KF32:
		base = "f32"
	case KF64:
		base = "f64"
	case KUnit:
		base = "()"
	case KString:
		base = "string"
	default:
		return "<invalid>"
	}

	if len(t.Shape) == 0 {
		return base
	}

	// Tensor type: f32[N, M]
	dims := make([]string, len(t.Shape))
	for i, d := range t.Shape {
		dims[i] = d.String()
	}
	return fmt.Sprintf("%s[%s]", base, strings.Join(dims, ", "))
}

// IsNumeric reports if the type supports arithmetic.
func (t *Type) IsNumeric() bool {
	switch t.K {
	case KI8, KI16, KI32, KI64, KU8, KU16, KU32, KU64, KF32, KF64:
		return true
	}
	return false
}

// IsInteger reports if the type is an integer.
func (t *Type) IsInteger() bool {
	switch t.K {
	case KI8, KI16, KI32, KI64, KU8, KU16, KU32, KU64:
		return true
	}
	return false
}

// IsUnsigned reports if the type is an unsigned integer. Useful for
// codegen dispatch (signed IDIV vs unsigned DIV, signed vs unsigned
// condition codes for comparisons).
func (t *Type) IsUnsigned() bool {
	if t == nil {
		return false
	}
	switch t.K {
	case KU8, KU16, KU32, KU64:
		return true
	}
	return false
}

// Equal reports whether two types are structurally equal.
func (t *Type) Equal(u *Type) bool {
	if t == nil || u == nil {
		return t == u
	}
	if t.K != u.K {
		return false
	}
	if len(t.Shape) != len(u.Shape) {
		return false
	}
	for i := range t.Shape {
		if !t.Shape[i].Equal(u.Shape[i]) {
			return false
		}
	}
	return true
}

// IsTensor reports if this is a tensor type (has shape).
func (t *Type) IsTensor() bool {
	return len(t.Shape) > 0
}

// Rank returns the number of dimensions (0 for scalars).
func (t *Type) Rank() int {
	return len(t.Shape)
}

// ElemType returns a scalar type with the same element kind.
func (t *Type) ElemType() *Type {
	return &Type{K: t.K}
}

// WithShape returns a tensor type with the given shape.
func (t *Type) WithShape(shape []ShapeDim) *Type {
	return &Type{K: t.K, Shape: shape}
}

// EffectSet is the set of effects a function may perform. v0.13
// introduces a single effect, `@io`; further effects (`@alloc`,
// `@panic`, `@nondet`) ride along the same bitmask in later
// milestones. The empty set means pure.
type EffectSet uint8

const (
	// EffIO is the I/O effect. Functions that touch stdout, files, or
	// any other observable side-channel must carry this effect.
	EffIO EffectSet = 1 << iota
)

// Has reports whether `e` includes effect `f`.
func (e EffectSet) Has(f EffectSet) bool { return e&f == f }

// Union returns the union of two effect sets.
func (e EffectSet) Union(f EffectSet) EffectSet { return e | f }

// IsPure reports whether the set is empty.
func (e EffectSet) IsPure() bool { return e == 0 }

// Subset reports whether every effect in `e` is also in `f`. A pure
// function (empty set) is a subset of every function.
func (e EffectSet) Subset(f EffectSet) bool { return e&^f == 0 }

// String renders the set as a space-separated `@name` list, or "" for
// the empty set. Order matches bit position (currently just `@io`).
func (e EffectSet) String() string {
	if e == 0 {
		return ""
	}
	var parts []string
	if e.Has(EffIO) {
		parts = append(parts, "@io")
	}
	return strings.Join(parts, " ")
}

// Lookup resolves a named type.
func Lookup(name string) *Type {
	switch name {
	case "bool":
		return Bool()
	case "i8":
		return I8()
	case "i16":
		return I16()
	case "i32":
		return I32()
	case "i64":
		return I64()
	case "u8":
		return U8()
	case "u16":
		return U16()
	case "u32":
		return U32()
	case "u64":
		return U64()
	case "f32":
		return F32()
	case "f64":
		return F64()
	case "string":
		return String()
	}
	return nil
}
