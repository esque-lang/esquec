// Package stdlib defines built-in functions for the esque language.
// These functions are recognized by the type checker and lowered
// directly to efficient MIR operations.
package stdlib

import "github.com/esque-lang/esquec/internal/types"

// Builtin represents a built-in function definition.
type Builtin struct {
	Name      string
	Params    []BuiltinParam
	RetType   func(args []*types.Type) *types.Type // computes return type from arg types
	IsPure    bool                                  // true if function has no side effects
	IsIntrinsic bool                                // true if lowered directly to instructions
}

// BuiltinParam describes a parameter of a built-in function.
type BuiltinParam struct {
	Name     string
	Type     *types.Type
	IsVector bool // true if this param can be a tensor
}

// All built-in functions indexed by name.
var Builtins = map[string]*Builtin{
	// Math functions (element-wise on tensors)
	"exp": {
		Name:        "exp",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"log": {
		Name:        "log",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"sqrt": {
		Name:        "sqrt",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"pow": {
		Name:        "pow",
		Params:      []BuiltinParam{{Name: "base", IsVector: true}, {Name: "exp", IsVector: true}},
		RetType:     firstType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"sin": {
		Name:        "sin",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"cos": {
		Name:        "cos",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"abs": {
		Name:        "abs",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"min": {
		Name:        "min",
		Params:      []BuiltinParam{{Name: "a", IsVector: true}, {Name: "b", IsVector: true}},
		RetType:     firstType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"max": {
		Name:        "max",
		Params:      []BuiltinParam{{Name: "a", IsVector: true}, {Name: "b", IsVector: true}},
		RetType:     firstType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"clamp": {
		Name:        "clamp",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}, {Name: "min", IsVector: true}, {Name: "max", IsVector: true}},
		RetType:     firstType,
		IsPure:      true,
		IsIntrinsic: true,
	},

	// Neural network activation functions
	"relu": {
		Name:        "relu",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"sigmoid": {
		Name:        "sigmoid",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"tanh": {
		Name:        "tanh",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"softmax": {
		Name:        "softmax",
		Params:      []BuiltinParam{{Name: "x", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: false, // requires reduction, not a simple intrinsic
	},

	// Graphics/shader functions
	"normalize": {
		Name:        "normalize",
		Params:      []BuiltinParam{{Name: "v", IsVector: true}},
		RetType:     sameType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"dot": {
		Name:        "dot",
		Params:      []BuiltinParam{{Name: "a", IsVector: true}, {Name: "b", IsVector: true}},
		RetType:     scalarType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"cross": {
		Name:        "cross",
		Params:      []BuiltinParam{{Name: "a", IsVector: true}, {Name: "b", IsVector: true}},
		RetType:     firstType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"mix": {
		Name:        "mix",
		Params:      []BuiltinParam{{Name: "a", IsVector: true}, {Name: "b", IsVector: true}, {Name: "t", IsVector: true}},
		RetType:     firstType,
		IsPure:      true,
		IsIntrinsic: true,
	},
	"smoothstep": {
		Name:        "smoothstep",
		Params:      []BuiltinParam{{Name: "edge0", IsVector: true}, {Name: "edge1", IsVector: true}, {Name: "x", IsVector: true}},
		RetType:     thirdType,
		IsPure:      true,
		IsIntrinsic: true,
	},

	// Tensor operations
	"transpose": {
		Name:        "transpose",
		Params:      []BuiltinParam{{Name: "m", IsVector: true}},
		RetType:     transposeType,
		IsPure:      true,
		IsIntrinsic: false,
	},
	"flatten": {
		Name:        "flatten",
		Params:      []BuiltinParam{{Name: "t", IsVector: true}},
		RetType:     flattenType,
		IsPure:      true,
		IsIntrinsic: false,
	},
}

// Return type functions

// sameType returns the same type as the first argument
func sameType(args []*types.Type) *types.Type {
	if len(args) == 0 {
		return nil
	}
	return args[0]
}

// firstType returns the type of the first argument
func firstType(args []*types.Type) *types.Type {
	if len(args) == 0 {
		return nil
	}
	return args[0]
}

// thirdType returns the type of the third argument
func thirdType(args []*types.Type) *types.Type {
	if len(args) < 3 {
		return nil
	}
	return args[2]
}

// scalarType returns the scalar version of the first argument's type
func scalarType(args []*types.Type) *types.Type {
	if len(args) == 0 {
		return nil
	}
	return args[0].ElemType()
}

// transposeType returns a transposed shape for matrices
func transposeType(args []*types.Type) *types.Type {
	if len(args) == 0 || len(args[0].Shape) != 2 {
		return args[0]
	}
	// Swap dimensions
	return &types.Type{
		K: args[0].K,
		Shape: []types.ShapeDim{
			args[0].Shape[1],
			args[0].Shape[0],
		},
	}
}

// flattenType returns a 1D tensor with the product of all dimensions
func flattenType(args []*types.Type) *types.Type {
	if len(args) == 0 {
		return nil
	}
	t := args[0]
	if !t.IsTensor() {
		return t
	}
	// Calculate total elements
	var total int64 = 1
	for _, d := range t.Shape {
		if d.Var != "" {
			// Variable dimension - can't compute static size
			return &types.Type{
				K:     t.K,
				Shape: []types.ShapeDim{types.VarDim("_")},
			}
		}
		total *= d.Val
	}
	return &types.Type{
		K:     t.K,
		Shape: []types.ShapeDim{types.ConcreteDim(total)},
	}
}

// Lookup returns the builtin function with the given name, or nil if not found.
func Lookup(name string) *Builtin {
	return Builtins[name]
}

// IsBuiltin returns true if the given name is a built-in function.
func IsBuiltin(name string) bool {
	_, ok := Builtins[name]
	return ok
}
