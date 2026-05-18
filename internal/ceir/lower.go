package ceir

import (
	"fmt"
	"sort"
	"strings"

	"github.com/esque-lang/esquec/internal/ast"
	"github.com/esque-lang/esquec/internal/diag"
	"github.com/esque-lang/esquec/internal/types"
)

// LowerFromChecked converts a typed AST file into a CEIR module.
func LowerFromChecked(file *types.CheckedFile) (*Module, error) {
	// Build function signature map for call lowering
	fnSigs := map[string]*types.FnSig{}
	// Map from original generic name to list of monomorphized function names
	monoVersions := map[string][]string{}

	// Runtime intrinsics: their bodies live in the backend's runtime
	// helpers, but call lowering still needs their signatures so it can
	// resolve return types when seeding OpCall instructions.
	fnSigs["print_i32"] = &types.FnSig{
		Params:  []*types.Type{types.I32()},
		RetType: types.I32(),
	}
	fnSigs["print_f32"] = &types.FnSig{
		Params:  []*types.Type{types.F32()},
		RetType: types.F32(),
	}
	fnSigs["print_str"] = &types.FnSig{
		Params:  []*types.Type{types.String()},
		RetType: types.String(),
	}

	for _, fn := range file.Fns {
		var shapeParams []string
		for _, sp := range fn.AST.ShapeParams {
			shapeParams = append(shapeParams, sp.Name)
		}
		var params []*types.Type
		for _, p := range fn.Params {
			params = append(params, p.Type)
		}
		fnSigs[fn.Name] = &types.FnSig{ShapeParams: shapeParams, Params: params, RetType: fn.RetType}

		// Track monomorphized versions for call resolution
		if fn.IsMonomorphized && fn.OriginalName != "" {
			monoVersions[fn.OriginalName] = append(monoVersions[fn.OriginalName], fn.Name)
		}
	}

	m := &Module{Path: file.Path}
	for _, fn := range file.Fns {
		cf, err := lowerFn(fn, fnSigs, monoVersions)
		if err != nil {
			return nil, err
		}
		m.Fns = append(m.Fns, cf)
	}
	return m, nil
}

type lowerCtx struct {
	nextID       int
	insts        []Inst
	bindings     map[string]Value
	fnSigs       map[string]*types.FnSig
	monoVersions map[string][]string   // original name -> list of monomorphized names
	shapeSubst   map[string]int64      // current shape substitutions (for monomorphized fns)
}

func (c *lowerCtx) fresh(t *types.Type) Value {
	v := Value{ID: c.nextID, Type: t}
	c.nextID++
	return v
}

func lowerFn(f *types.CheckedFn, fnSigs map[string]*types.FnSig, monoVersions map[string][]string) (*Func, error) {
	c := &lowerCtx{
		bindings:     map[string]Value{},
		fnSigs:       fnSigs,
		monoVersions: monoVersions,
		shapeSubst:   f.ShapeValues, // nil for non-monomorphized functions
	}
	var params []Param
	for _, p := range f.Params {
		v := c.fresh(p.Type)
		c.bindings[p.Name] = v
		params = append(params, Param{Name: p.Name, Type: p.Type})
	}
	res, err := c.lowerExpr(f.AST.Body)
	if err != nil {
		return nil, err
	}
	return &Func{
		Name:    f.Name,
		Params:  params,
		RetType: f.RetType,
		Body:    c.insts,
		Result:  res,
	}, nil
}

// computeInstKey computes the monomorphized function name given shape bindings.
func computeInstKey(fnName string, shapeValues map[string]int64) string {
	if len(shapeValues) == 0 {
		return fnName
	}
	var keys []string
	for k := range shapeValues {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%d", shapeValues[k]))
	}
	// Use __ as separator to avoid issues with $ in symbol names
	return fmt.Sprintf("%s__%s", fnName, strings.Join(parts, "_"))
}

// inferShapeArgsFromSig infers shape arguments from argument types using the signature.
func (c *lowerCtx) inferShapeArgsFromSig(sig *types.FnSig, argTypes []*types.Type) map[string]int64 {
	if len(sig.ShapeParams) == 0 {
		return nil
	}

	result := make(map[string]int64)
	for i, argType := range argTypes {
		if i >= len(sig.Params) {
			break
		}
		paramType := sig.Params[i]
		c.unifyShapes(paramType, argType, result)
	}

	// Check we got all shape params
	for _, sp := range sig.ShapeParams {
		if _, ok := result[sp]; !ok {
			return nil
		}
	}

	return result
}

// unifyShapes extracts shape bindings by matching param type with arg type.
func (c *lowerCtx) unifyShapes(paramType, argType *types.Type, bindings map[string]int64) {
	if !paramType.IsTensor() || !argType.IsTensor() {
		return
	}
	if len(paramType.Shape) != len(argType.Shape) {
		return
	}
	for i := range paramType.Shape {
		pDim := paramType.Shape[i]
		aDim := argType.Shape[i]
		if pDim.Var != "" && aDim.Var == "" {
			bindings[pDim.Var] = aDim.Val
		}
	}
}

// findMatchingMonoVersion finds the monomorphized version that matches the given argument types.
func (c *lowerCtx) findMatchingMonoVersion(origName string, versions []string, argTypes []*types.Type) string {
	for _, ver := range versions {
		sig := c.fnSigs[ver]
		if sig == nil || len(sig.Params) != len(argTypes) {
			continue
		}
		match := true
		for i, param := range sig.Params {
			if !c.typesMatch(param, argTypes[i]) {
				match = false
				break
			}
		}
		if match {
			return ver
		}
	}
	// No matching version found, fall back to original name
	return origName
}

// typesMatch checks if two types are compatible (same kind, same shape).
func (c *lowerCtx) typesMatch(a, b *types.Type) bool {
	if a.K != b.K {
		return false
	}
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		// Compare concrete values (both should be concrete at this point)
		if a.Shape[i].Val != b.Shape[i].Val {
			return false
		}
	}
	return true
}

func (c *lowerCtx) lowerExpr(e ast.Expr) (Value, error) {
	switch n := e.(type) {
	case *ast.IntLit:
		// Honour an explicit type suffix when present. The type checker
		// has already validated suffix correctness and value range, so
		// here we just dispatch on the suffix string. An empty suffix
		// defaults to i32, matching the type checker.
		var t *types.Type
		switch n.TypeSuffix {
		case "":
			t = types.I32()
		case "i8":
			t = types.I8()
		case "i32":
			t = types.I32()
		case "i64":
			t = types.I64()
		case "u8":
			t = types.U8()
		case "u32":
			t = types.U32()
		default:
			// Type checker should have rejected unsupported suffixes
			// before we got here; treat anything else as i32 to keep
			// lowering total.
			t = types.I32()
		}
		v := c.fresh(t)
		c.insts = append(c.insts, Inst{Result: v, Op: OpConstInt, Imm: n.Value})
		return v, nil

	case *ast.FloatLit:
		var t *types.Type
		if n.Width == 64 {
			t = types.F64()
		} else {
			t = types.F32()
		}
		v := c.fresh(t)
		c.insts = append(c.insts, Inst{Result: v, Op: OpConstFloat, ImmF: n.Value})
		return v, nil

	case *ast.BoolLit:
		v := c.fresh(types.Bool())
		c.insts = append(c.insts, Inst{Result: v, Op: OpConstBool, ImmB: n.Value})
		return v, nil

	case *ast.StringLit:
		// Strings are fat pointers (ptr, len). At CEIR time we keep
		// the value single-Value-shaped: ImmS holds the bytes, Imm
		// holds the length. The MIR lowering interns the bytes in
		// .rodata and tracks the length in a side-table so the
		// per-arg call lowering can re-emit (ptr, len) at use sites.
		v := c.fresh(types.String())
		c.insts = append(c.insts, Inst{
			Result: v,
			Op:     OpConstString,
			ImmS:   n.Value,
			Imm:    int64(len(n.Value)),
		})
		return v, nil

	case *ast.Ident:
		v, ok := c.bindings[n.Name]
		if !ok {
			return Value{}, diag.Errorf(n.Span(), "ceir: unbound identifier %q", n.Name)
		}
		return v, nil

	case *ast.BinOp:
		l, err := c.lowerExpr(n.L)
		if err != nil {
			return Value{}, err
		}
		r, err := c.lowerExpr(n.R)
		if err != nil {
			return Value{}, err
		}
		op, resType := binOpToOp(n.Op, l.Type, r.Type)
		v := c.fresh(resType)
		c.insts = append(c.insts, Inst{Result: v, Op: op, Args: []Value{l, r}})
		return v, nil

	case *ast.Unary:
		x, err := c.lowerExpr(n.X)
		if err != nil {
			return Value{}, err
		}
		var op Op
		switch n.Op {
		case "-":
			op = OpNeg
		case "!":
			op = OpNot
		default:
			return Value{}, diag.Errorf(n.Span(), "ceir: unsupported unary %q", n.Op)
		}
		v := c.fresh(x.Type)
		c.insts = append(c.insts, Inst{Result: v, Op: op, Args: []Value{x}})
		return v, nil

	case *ast.Call:
		ident, ok := n.Fn.(*ast.Ident)
		if !ok {
			return Value{}, diag.Errorf(n.Fn.Span(), "ceir: callee must be identifier")
		}
		var args []Value
		var argTypes []*types.Type
		for _, arg := range n.Args {
			a, err := c.lowerExpr(arg)
			if err != nil {
				return Value{}, err
			}
			args = append(args, a)
			argTypes = append(argTypes, a.Type)
		}
		// Determine the actual function name (may be monomorphized)
		fnName := ident.Name
		// Check if this is a generic function with monomorphized versions
		if versions, ok := c.monoVersions[ident.Name]; ok {
			// Find the version that matches our argument types
			fnName = c.findMatchingMonoVersion(ident.Name, versions, argTypes)
		}
		// Get result type from the resolved function signature
		resultSig := c.fnSigs[fnName]
		if resultSig == nil {
			return Value{}, diag.Errorf(ident.Span(), "ceir: unknown function %q", fnName)
		}
		v := c.fresh(resultSig.RetType)
		c.insts = append(c.insts, Inst{Result: v, Op: OpCall, FnName: fnName, Args: args})
		return v, nil

	case *ast.Block:
		// Save current bindings and restore after
		oldBindings := make(map[string]Value, len(c.bindings))
		for k, v := range c.bindings {
			oldBindings[k] = v
		}
		defer func() { c.bindings = oldBindings }()

		for _, stmt := range n.Stmts {
			if err := c.lowerStmt(stmt); err != nil {
				return Value{}, err
			}
		}
		if n.Result == nil {
			// Unit result
			v := c.fresh(types.Unit())
			return v, nil
		}
		return c.lowerExpr(n.Result)

	case *ast.If:
		cond, err := c.lowerExpr(n.Cond)
		if err != nil {
			return Value{}, err
		}
		thenV, err := c.lowerExpr(n.Then)
		if err != nil {
			return Value{}, err
		}
		if n.Else == nil {
			// No else: result is unit
			v := c.fresh(types.Unit())
			return v, nil
		}
		elseV, err := c.lowerExpr(n.Else)
		if err != nil {
			return Value{}, err
		}
		// Use select instruction: result = select(cond, thenV, elseV)
		v := c.fresh(thenV.Type)
		c.insts = append(c.insts, Inst{Result: v, Op: OpSelect, Args: []Value{cond, thenV, elseV}})
		return v, nil

	case *ast.Pipeline:
		// x |> f desugars to f(x)
		// x |> f(a) desugars to f(x, a)
		lhs, err := c.lowerExpr(n.L)
		if err != nil {
			return Value{}, err
		}
		switch rhs := n.R.(type) {
		case *ast.Ident:
			fnName := rhs.Name
			argTypes := []*types.Type{lhs.Type}
			sig := c.fnSigs[rhs.Name]
			if sig != nil && len(sig.ShapeParams) > 0 {
				shapeArgs := c.inferShapeArgsFromSig(sig, argTypes)
				if shapeArgs != nil {
					fnName = computeInstKey(rhs.Name, shapeArgs)
				}
			}
			resultSig := c.fnSigs[fnName]
			if resultSig == nil {
				resultSig = sig
			}
			v := c.fresh(resultSig.RetType)
			c.insts = append(c.insts, Inst{Result: v, Op: OpCall, FnName: fnName, Args: []Value{lhs}})
			return v, nil
		case *ast.Call:
			ident, _ := rhs.Fn.(*ast.Ident)
			args := []Value{lhs}
			argTypes := []*types.Type{lhs.Type}
			for _, arg := range rhs.Args {
				a, err := c.lowerExpr(arg)
				if err != nil {
					return Value{}, err
				}
				args = append(args, a)
				argTypes = append(argTypes, a.Type)
			}
			fnName := ident.Name
			sig := c.fnSigs[ident.Name]
			if sig != nil && len(sig.ShapeParams) > 0 {
				shapeArgs := c.inferShapeArgsFromSig(sig, argTypes)
				if shapeArgs != nil {
					fnName = computeInstKey(ident.Name, shapeArgs)
				}
			}
			resultSig := c.fnSigs[fnName]
			if resultSig == nil {
				resultSig = sig
			}
			v := c.fresh(resultSig.RetType)
			c.insts = append(c.insts, Inst{Result: v, Op: OpCall, FnName: fnName, Args: args})
			return v, nil
		default:
			return Value{}, diag.Errorf(n.R.Span(), "ceir: pipeline rhs must be ident or call")
		}

	case *ast.TensorLit:
		// Lower each element
		var elemVals []Value
		for _, elem := range n.Elems {
			ev, err := c.lowerExpr(elem)
			if err != nil {
				return Value{}, err
			}
			elemVals = append(elemVals, ev)
		}
		// Create tensor type with shape
		elemType := elemVals[0].Type
		// Check if elements are tensors (rank-2+ literal)
		if elemType.IsTensor() {
			// e.g., [[1,2],[3,4]] → elements are f32[2], result is f32[2,2]
			outerDim := types.ConcreteDim(int64(len(elemVals)))
			resultShape := append([]types.ShapeDim{outerDim}, elemType.Shape...)
			tensorType := elemType.ElemType().WithShape(resultShape)
			v := c.fresh(tensorType)
			c.insts = append(c.insts, Inst{Result: v, Op: OpTensorLit, Args: elemVals})
			return v, nil
		}
		// Rank-1 case
		shape := []types.ShapeDim{types.ConcreteDim(int64(len(elemVals)))}
		tensorType := elemType.WithShape(shape)
		v := c.fresh(tensorType)
		c.insts = append(c.insts, Inst{Result: v, Op: OpTensorLit, Args: elemVals})
		return v, nil

	case *ast.Reduce:
		x, err := c.lowerExpr(n.X)
		if err != nil {
			return Value{}, err
		}
		// Check if operand is a tensor
		if x.Type.IsTensor() {
			// Integer reductions: the MIR/x86 reduce path is f32-specific
			// (it uses haddps/addps), so for i32 tensors we expand the
			// reduction inline as a scalar OpAdd/OpMul chain. Rank-1 only;
			// higher-rank integer reductions remain unsupported in v0.10.
			if x.Type.Rank() == 1 && x.Type.ElemType().K == types.KI32 {
				count := x.Type.Shape[0].Val
				if count <= 0 {
					return Value{}, diag.Errorf(n.X.Span(), "ceir: reduce requires concrete tensor size")
				}
				elemType := x.Type.ElemType()
				var binOp Op
				var acc Value
				var startIdx int64
				switch n.Op {
				case "+":
					binOp = OpAdd
					acc = c.fresh(elemType)
					c.insts = append(c.insts, Inst{Result: acc, Op: OpConstInt, Imm: 0})
				case "*":
					binOp = OpMul
					acc = c.fresh(elemType)
					c.insts = append(c.insts, Inst{Result: acc, Op: OpConstInt, Imm: 1})
				default:
					return Value{}, diag.Errorf(n.Span(), "ceir: unsupported integer reduction %q", n.Op)
				}
				for i := startIdx; i < count; i++ {
					elem := c.fresh(elemType)
					c.insts = append(c.insts, Inst{Result: elem, Op: OpTensorElem, Args: []Value{x}, Imm: i})
					newAcc := c.fresh(elemType)
					c.insts = append(c.insts, Inst{Result: newAcc, Op: binOp, Args: []Value{acc, elem}})
					acc = newAcc
				}
				return acc, nil
			}

			// Generate reduction operation
			var op Op
			switch n.Op {
			case "+":
				op = OpReduceSum
			case "*":
				op = OpReduceProd
			default:
				return Value{}, diag.Errorf(n.Span(), "ceir: unsupported reduction operator %q", n.Op)
			}
			// Result type depends on rank:
			// - Rank-1 reduces to scalar
			// - Rank-2+ reduces last dimension (e.g., f32[M,N] -> f32[M])
			var resultType *types.Type
			if x.Type.Rank() == 1 {
				resultType = x.Type.ElemType()
			} else {
				// Higher rank: remove last dimension
				newShape := make([]types.ShapeDim, x.Type.Rank()-1)
				copy(newShape, x.Type.Shape[:x.Type.Rank()-1])
				resultType = x.Type.ElemType().WithShape(newShape)
			}
			v := c.fresh(resultType)
			c.insts = append(c.insts, Inst{Result: v, Op: op, Args: []Value{x}})
			return v, nil
		}
		// For scalars, reduction is identity
		v := c.fresh(x.Type)
		c.insts = append(c.insts, Inst{Result: v, Op: OpCopy, Args: []Value{x}})
		return v, nil

	case *ast.Cast:
		// Lower type cast
		x, err := c.lowerExpr(n.X)
		if err != nil {
			return Value{}, err
		}
		// Determine cast operation based on source and target types
		// We need to resolve the target type from AST
		targetName := ""
		if named, ok := n.Target.(*ast.NamedType); ok {
			targetName = named.Name
		}
		var op Op
		var resultType *types.Type
		switch {
		case x.Type.K == types.KF32 && targetName == "i32":
			op = OpCastF32ToI32
			resultType = types.I32()
		case x.Type.K == types.KF64 && targetName == "i32":
			op = OpCastF64ToI32
			resultType = types.I32()
		case x.Type.K == types.KI32 && targetName == "f32":
			op = OpCastI32ToF32
			resultType = types.F32()
		case x.Type.K == types.KI32 && targetName == "f64":
			op = OpCastI32ToF64
			resultType = types.F64()

		// i8/u8 widening casts.
		case x.Type.K == types.KI8 && targetName == "i32":
			op = OpCastI8ToI32
			resultType = types.I32()
		case x.Type.K == types.KI8 && targetName == "i64":
			op = OpCastI8ToI64
			resultType = types.I64()
		case x.Type.K == types.KU8 && targetName == "i32":
			op = OpCastU8ToI32
			resultType = types.I32()
		case x.Type.K == types.KU8 && targetName == "i64":
			op = OpCastU8ToI64
			resultType = types.I64()

		// Truncating casts to i8/u8.
		case x.Type.K == types.KI32 && targetName == "i8":
			op = OpCastI32ToI8
			resultType = types.I8()
		case x.Type.K == types.KI32 && targetName == "u8":
			op = OpCastI32ToU8
			resultType = types.U8()
		case x.Type.K == types.KI64 && targetName == "i8":
			op = OpCastI64ToI8
			resultType = types.I8()
		case x.Type.K == types.KI64 && targetName == "u8":
			op = OpCastI64ToU8
			resultType = types.U8()
		case x.Type.K == types.KU32 && targetName == "i8":
			// u32 already lives in the low 32 bits of a 64-bit GPR;
			// truncate by reusing the i32→i8 path.
			op = OpCastI32ToI8
			resultType = types.I8()
		case x.Type.K == types.KU32 && targetName == "u8":
			op = OpCastI32ToU8
			resultType = types.U8()

		// i8↔u8 reinterpretation: emit a re-canonicalise step so the
		// destination type's invariant holds. i8→u8 zero-extends the
		// low byte; u8→i8 sign-extends it.
		case x.Type.K == types.KI8 && targetName == "u8":
			op = OpCastI32ToU8
			resultType = types.U8()
		case x.Type.K == types.KU8 && targetName == "i8":
			op = OpCastI32ToI8
			resultType = types.I8()

		default:
			return Value{}, diag.Errorf(n.Span(), "ceir: unsupported cast from %s to %s", x.Type, targetName)
		}
		v := c.fresh(resultType)
		c.insts = append(c.insts, Inst{Result: v, Op: op, Args: []Value{x}})
		return v, nil

	case *ast.Match:
		// Lower match to a chain of comparisons and selects
		scrutinee, err := c.lowerExpr(n.Scrutinee)
		if err != nil {
			return Value{}, err
		}

		if len(n.Arms) == 0 {
			return Value{}, diag.Errorf(n.Span(), "ceir: match must have at least one arm")
		}

		// Lower each arm's body first (we need all values for selects)
		type armInfo struct {
			cond Value // the condition to check (or empty for wildcard)
			body Value // the body value
		}
		var arms []armInfo

		for _, arm := range n.Arms {
			var cond Value

			switch pat := arm.Pattern.(type) {
			case *ast.WildcardPat:
				// Wildcard always matches - cond is true
				cond = c.fresh(types.Bool())
				c.insts = append(c.insts, Inst{Result: cond, Op: OpConstBool, ImmB: true})

			case *ast.BindPat:
				// Binding pattern always matches - bind the value and cond is true
				c.bindings[pat.Name] = scrutinee
				cond = c.fresh(types.Bool())
				c.insts = append(c.insts, Inst{Result: cond, Op: OpConstBool, ImmB: true})

			case *ast.LitPat:
				// Literal pattern - compare scrutinee to literal
				litVal, err := c.lowerExpr(pat.Value)
				if err != nil {
					return Value{}, err
				}
				cond = c.fresh(types.Bool())
				c.insts = append(c.insts, Inst{Result: cond, Op: OpEq, Args: []Value{scrutinee, litVal}})
			}

			// Handle guard if present
			if arm.Guard != nil {
				guardVal, err := c.lowerExpr(arm.Guard)
				if err != nil {
					return Value{}, err
				}
				// cond = cond && guard
				newCond := c.fresh(types.Bool())
				c.insts = append(c.insts, Inst{Result: newCond, Op: OpAnd, Args: []Value{cond, guardVal}})
				cond = newCond
			}

			// Lower body
			body, err := c.lowerExpr(arm.Body)
			if err != nil {
				return Value{}, err
			}

			arms = append(arms, armInfo{cond: cond, body: body})
		}

		// Now chain the arms together with selects, from last to first
		// result = select(cond[n-1], body[n-1], select(cond[n-2], body[n-2], ...))
		result := arms[len(arms)-1].body
		for i := len(arms) - 2; i >= 0; i-- {
			newResult := c.fresh(result.Type)
			c.insts = append(c.insts, Inst{
				Result: newResult,
				Op:     OpSelect,
				Args:   []Value{arms[i].cond, arms[i].body, result},
			})
			result = newResult
		}

		return result, nil

	case *ast.Lambda:
		// Lambdas are handled at their use site (in Fold, Iterate, etc.)
		// Standalone lambdas are not supported in CEIR
		return Value{}, diag.Errorf(n.Span(), "ceir: standalone lambda not supported; use in fold or iterate context")

	case *ast.Fold:
		// Lower fold expression: f/tensor or f/init/tensor
		// For rank-1 tensors, we inline the reduction
		return c.lowerFold(n)

	case *ast.Iterate:
		// Lower iterate(n, init, f) by unrolling the loop
		return c.lowerIterate(n)

	case *ast.RangeExpr:
		return c.lowerRange(n)

	case *ast.Tabulate:
		return c.lowerTabulate(n)

	case *ast.Scan:
		return c.lowerScan(n)

	case *ast.IterateUntil:
		return c.lowerIterateUntil(n)

	case *ast.Each:
		return c.lowerEach(n)
	}

	return Value{}, diag.Errorf(e.Span(), "ceir: unsupported expression %T", e)
}

// lowerFold handles fold expressions: f/tensor or f/init/tensor
// For lambdas, it inlines the lambda body for each reduction step.
func (c *lowerCtx) lowerFold(n *ast.Fold) (Value, error) {
	x, err := c.lowerExpr(n.X)
	if err != nil {
		return Value{}, err
	}

	if !x.Type.IsTensor() {
		return Value{}, diag.Errorf(n.X.Span(), "ceir: fold requires tensor operand, got %s", x.Type)
	}

	// Get tensor size for rank-1 tensors
	if x.Type.Rank() != 1 {
		return Value{}, diag.Errorf(n.X.Span(), "ceir: fold currently only supports rank-1 tensors, got rank-%d", x.Type.Rank())
	}
	tensorLen := x.Type.Shape[0].Val
	if tensorLen <= 0 {
		return Value{}, diag.Errorf(n.X.Span(), "ceir: fold requires concrete tensor size")
	}

	elemType := x.Type.ElemType()

	// Check if it's a lambda or named function
	if lambda, ok := n.Fn.(*ast.Lambda); ok {
		// Lambda fold: inline the lambda body
		if len(lambda.Params) != 2 {
			return Value{}, diag.Errorf(lambda.Span(), "ceir: fold lambda must have exactly 2 parameters, got %d", len(lambda.Params))
		}

		// Determine initial accumulator
		var acc Value
		startIdx := int64(0)
		if n.Init != nil {
			// f/init/tensor - use provided init
			acc, err = c.lowerExpr(n.Init)
			if err != nil {
				return Value{}, err
			}
		} else {
			// f/tensor - use first element as init
			acc = c.fresh(elemType)
			c.insts = append(c.insts, Inst{Result: acc, Op: OpTensorElem, Args: []Value{x}, Imm: 0})
			startIdx = 1
		}

		// Inline the lambda for each element
		for i := startIdx; i < tensorLen; i++ {
			// Load element i
			elem := c.fresh(elemType)
			c.insts = append(c.insts, Inst{Result: elem, Op: OpTensorElem, Args: []Value{x}, Imm: i})

			// Bind lambda parameters
			oldBindings := make(map[string]Value)
			oldBindings[lambda.Params[0].Name] = c.bindings[lambda.Params[0].Name]
			oldBindings[lambda.Params[1].Name] = c.bindings[lambda.Params[1].Name]
			c.bindings[lambda.Params[0].Name] = acc
			c.bindings[lambda.Params[1].Name] = elem

			// Lower lambda body
			newAcc, err := c.lowerExpr(lambda.Body)
			if err != nil {
				return Value{}, err
			}

			// Restore bindings
			if v, ok := oldBindings[lambda.Params[0].Name]; ok {
				c.bindings[lambda.Params[0].Name] = v
			} else {
				delete(c.bindings, lambda.Params[0].Name)
			}
			if v, ok := oldBindings[lambda.Params[1].Name]; ok {
				c.bindings[lambda.Params[1].Name] = v
			} else {
				delete(c.bindings, lambda.Params[1].Name)
			}

			acc = newAcc
		}

		return acc, nil
	}

	// Named function fold
	ident, ok := n.Fn.(*ast.Ident)
	if !ok {
		return Value{}, diag.Errorf(n.Fn.Span(), "ceir: fold function must be identifier or lambda")
	}
	fnName := ident.Name

	// Check for built-in operators that map to existing reduce operations
	if fnName == "+" || fnName == "add" {
		v := c.fresh(elemType)
		c.insts = append(c.insts, Inst{Result: v, Op: OpReduceSum, Args: []Value{x}})
		return v, nil
	}
	if fnName == "*" || fnName == "mul" {
		v := c.fresh(elemType)
		c.insts = append(c.insts, Inst{Result: v, Op: OpReduceProd, Args: []Value{x}})
		return v, nil
	}
	if fnName == "max" {
		v := c.fresh(elemType)
		c.insts = append(c.insts, Inst{Result: v, Op: OpReduceMax, Args: []Value{x}})
		return v, nil
	}
	if fnName == "min" {
		v := c.fresh(elemType)
		c.insts = append(c.insts, Inst{Result: v, Op: OpReduceMin, Args: []Value{x}})
		return v, nil
	}

	// For other named functions, call them in a loop
	sig, ok := c.fnSigs[fnName]
	if !ok {
		return Value{}, diag.Errorf(ident.Span(), "ceir: unknown fold function %q", fnName)
	}
	if len(sig.Params) != 2 {
		return Value{}, diag.Errorf(ident.Span(), "ceir: fold function %q must have 2 parameters, has %d", fnName, len(sig.Params))
	}

	// Determine initial accumulator
	var acc Value
	startIdx := int64(0)
	if n.Init != nil {
		acc, err = c.lowerExpr(n.Init)
		if err != nil {
			return Value{}, err
		}
	} else {
		acc = c.fresh(elemType)
		c.insts = append(c.insts, Inst{Result: acc, Op: OpTensorElem, Args: []Value{x}, Imm: 0})
		startIdx = 1
	}

	// Call function for each element
	for i := startIdx; i < tensorLen; i++ {
		elem := c.fresh(elemType)
		c.insts = append(c.insts, Inst{Result: elem, Op: OpTensorElem, Args: []Value{x}, Imm: i})

		newAcc := c.fresh(sig.RetType)
		c.insts = append(c.insts, Inst{Result: newAcc, Op: OpCall, FnName: fnName, Args: []Value{acc, elem}})
		acc = newAcc
	}

	return acc, nil
}

// lowerIterate handles iterate(n, init, f).
// For small counts (<= 32), it unrolls the loop by inlining the lambda body.
// For large counts, it emits OpIterateLoop which is lowered to an actual loop in MIR.
func (c *lowerCtx) lowerIterate(n *ast.Iterate) (Value, error) {
	count, err := types.EvalConstInt(n.N)
	if err != nil {
		return Value{}, err
	}
	if count <= 0 {
		return Value{}, diag.Errorf(n.N.Span(), "ceir: iterate requires positive constant count, got %d", count)
	}

	// Lower init value
	state, err := c.lowerExpr(n.Init)
	if err != nil {
		return Value{}, err
	}

	// Check if it's a lambda or named function
	if lambda, ok := n.Fn.(*ast.Lambda); ok {
		// Lambda iterate: inline the lambda body n times
		if len(lambda.Params) != 1 {
			return Value{}, diag.Errorf(lambda.Span(), "ceir: iterate lambda must have exactly 1 parameter, got %d", len(lambda.Params))
		}

		paramName := lambda.Params[0].Name

		// Threshold for unrolling vs loop
		const unrollThreshold = 32

		if count <= unrollThreshold {
			// Small count: unroll the iteration
			for i := int64(0); i < count; i++ {
				// Bind lambda parameter to current state
				oldBinding, hadOld := c.bindings[paramName]
				c.bindings[paramName] = state

				// Lower lambda body
				newState, err := c.lowerExpr(lambda.Body)
				if err != nil {
					return Value{}, err
				}

				// Restore binding
				if hadOld {
					c.bindings[paramName] = oldBinding
				} else {
					delete(c.bindings, paramName)
				}

				state = newState
			}
			return state, nil
		}

		// Large count: emit loop-based iteration
		// Lower the lambda body ONCE to capture the instructions
		loopCtx := &lowerCtx{
			nextID:       c.nextID,
			bindings:     make(map[string]Value),
			fnSigs:       c.fnSigs,
			monoVersions: c.monoVersions,
			shapeSubst:   c.shapeSubst,
		}

		// Copy outer bindings (for captured variables like 'b' in |x| x .+ b)
		for k, v := range c.bindings {
			loopCtx.bindings[k] = v
		}

		// Create a fresh value for the loop input (will be bound to state on each iteration)
		loopIn := loopCtx.fresh(state.Type)
		loopCtx.bindings[paramName] = loopIn

		// Lower the lambda body
		loopOut, err := loopCtx.lowerExpr(lambda.Body)
		if err != nil {
			return Value{}, err
		}

		// Update our nextID to account for the body instructions
		c.nextID = loopCtx.nextID

		// The result has the same type as the input state
		result := c.fresh(state.Type)

		// Emit the loop instruction
		c.insts = append(c.insts, Inst{
			Result:    result,
			Op:        OpIterateLoop,
			Imm:       count,
			Args:      []Value{state},
			BodyInsts: loopCtx.insts,
			LoopInID:  loopIn.ID,
			LoopOutID: loopOut.ID,
		})

		return result, nil
	}

	// Named function iterate
	ident, ok := n.Fn.(*ast.Ident)
	if !ok {
		return Value{}, diag.Errorf(n.Fn.Span(), "ceir: iterate function must be identifier or lambda")
	}
	fnName := ident.Name

	sig, ok := c.fnSigs[fnName]
	if !ok {
		return Value{}, diag.Errorf(ident.Span(), "ceir: unknown iterate function %q", fnName)
	}
	if len(sig.Params) != 1 {
		return Value{}, diag.Errorf(ident.Span(), "ceir: iterate function %q must have 1 parameter, has %d", fnName, len(sig.Params))
	}

	// For named functions, always emit loop for large counts
	const unrollThreshold = 32

	if count <= unrollThreshold {
		// Unroll the iteration with function calls
		for i := int64(0); i < count; i++ {
			newState := c.fresh(sig.RetType)
			c.insts = append(c.insts, Inst{Result: newState, Op: OpCall, FnName: fnName, Args: []Value{state}})
			state = newState
		}
		return state, nil
	}

	// Large count: emit OpIterate (will be lowered to loop in MIR)
	result := c.fresh(sig.RetType)
	c.insts = append(c.insts, Inst{
		Result: result,
		Op:     OpIterate,
		Imm:    count,
		Args:   []Value{state},
		FnName: fnName,
	})
	return result, nil
}

// applyUnaryFn applies a 1-arg lambda or named function to arg, emitting
// the necessary CEIR instructions. Lambdas are inlined; named functions
// emit OpCall.
func (c *lowerCtx) applyUnaryFn(fn ast.Expr, arg Value) (Value, error) {
	if lambda, ok := fn.(*ast.Lambda); ok {
		if len(lambda.Params) != 1 {
			return Value{}, diag.Errorf(lambda.Span(), "ceir: 1-arg lambda required, got %d params", len(lambda.Params))
		}
		paramName := lambda.Params[0].Name
		oldBinding, hadOld := c.bindings[paramName]
		c.bindings[paramName] = arg
		result, err := c.lowerExpr(lambda.Body)
		if hadOld {
			c.bindings[paramName] = oldBinding
		} else {
			delete(c.bindings, paramName)
		}
		return result, err
	}
	ident, ok := fn.(*ast.Ident)
	if !ok {
		return Value{}, diag.Errorf(fn.Span(), "ceir: function must be lambda or identifier")
	}
	sig, ok := c.fnSigs[ident.Name]
	if !ok {
		return Value{}, diag.Errorf(ident.Span(), "ceir: unknown function %q", ident.Name)
	}
	v := c.fresh(sig.RetType)
	c.insts = append(c.insts, Inst{Result: v, Op: OpCall, FnName: ident.Name, Args: []Value{arg}})
	return v, nil
}

// applyBinaryFn applies a 2-arg lambda or named function to (a, b).
func (c *lowerCtx) applyBinaryFn(fn ast.Expr, a, b Value) (Value, error) {
	if lambda, ok := fn.(*ast.Lambda); ok {
		if len(lambda.Params) != 2 {
			return Value{}, diag.Errorf(lambda.Span(), "ceir: 2-arg lambda required, got %d params", len(lambda.Params))
		}
		p0 := lambda.Params[0].Name
		p1 := lambda.Params[1].Name
		oldA, hadA := c.bindings[p0]
		oldB, hadB := c.bindings[p1]
		c.bindings[p0] = a
		c.bindings[p1] = b
		result, err := c.lowerExpr(lambda.Body)
		if hadA {
			c.bindings[p0] = oldA
		} else {
			delete(c.bindings, p0)
		}
		if hadB {
			c.bindings[p1] = oldB
		} else {
			delete(c.bindings, p1)
		}
		return result, err
	}
	ident, ok := fn.(*ast.Ident)
	if !ok {
		return Value{}, diag.Errorf(fn.Span(), "ceir: function must be lambda or identifier")
	}
	sig, ok := c.fnSigs[ident.Name]
	if !ok {
		return Value{}, diag.Errorf(ident.Span(), "ceir: unknown function %q", ident.Name)
	}
	v := c.fresh(sig.RetType)
	c.insts = append(c.insts, Inst{Result: v, Op: OpCall, FnName: ident.Name, Args: []Value{a, b}})
	return v, nil
}

// lowerRange lowers lo..hi or lo..=hi to an i32 iota tensor literal.
// The TensorLit will route through the existing rodata path in mir.
func (c *lowerCtx) lowerRange(n *ast.RangeExpr) (Value, error) {
	lo, err := types.EvalConstInt(n.Lo)
	if err != nil {
		return Value{}, err
	}
	hi, err := types.EvalConstInt(n.Hi)
	if err != nil {
		return Value{}, err
	}
	end := hi
	if n.Inclusive {
		end++
	}
	count := end - lo
	if count <= 0 {
		return Value{}, diag.Errorf(n.SpanRng, "ceir: empty or reversed range")
	}
	elems := make([]Value, 0, count)
	for v := lo; v < end; v++ {
		ev := c.fresh(types.I32())
		c.insts = append(c.insts, Inst{Result: ev, Op: OpConstInt, Imm: v})
		elems = append(elems, ev)
	}
	rt := types.I32().WithShape([]types.ShapeDim{types.ConcreteDim(count)})
	res := c.fresh(rt)
	c.insts = append(c.insts, Inst{Result: res, Op: OpTensorLit, Args: elems})
	return res, nil
}

// lowerTabulate lowers tabulate(N, |i| f(i)).
// For small counts (<= 32) the lambda body is unrolled N times and the
// per-iteration values are gathered into an OpTensorLit. For larger
// counts, an OpTabulateLoop is emitted so MIR can lower it to a
// counted runtime loop.
func (c *lowerCtx) lowerTabulate(n *ast.Tabulate) (Value, error) {
	count, err := types.EvalConstInt(n.N)
	if err != nil {
		return Value{}, err
	}
	if count <= 0 {
		return Value{}, diag.Errorf(n.N.Span(), "ceir: tabulate requires positive count, got %d", count)
	}

	const unrollThreshold = 32
	if count <= unrollThreshold {
		outputs := make([]Value, 0, count)
		var elemType *types.Type
		for i := int64(0); i < count; i++ {
			idxV := c.fresh(types.I32())
			c.insts = append(c.insts, Inst{Result: idxV, Op: OpConstInt, Imm: i})

			result, err := c.applyUnaryFn(n.Fn, idxV)
			if err != nil {
				return Value{}, err
			}
			if elemType == nil {
				elemType = result.Type
			}
			outputs = append(outputs, result)
		}
		rt := elemType.WithShape([]types.ShapeDim{types.ConcreteDim(count)})
		res := c.fresh(rt)
		c.insts = append(c.insts, Inst{Result: res, Op: OpTensorLit, Args: outputs})
		return res, nil
	}

	// Large count: lower the lambda body once into a sub-context to
	// capture BodyInsts, and emit OpTabulateLoop. Only lambdas are
	// supported on this path.
	lambda, ok := n.Fn.(*ast.Lambda)
	if !ok {
		return Value{}, diag.Errorf(n.Fn.Span(),
			"ceir: large-N tabulate (count %d) requires a lambda function", count)
	}
	if len(lambda.Params) != 1 {
		return Value{}, diag.Errorf(lambda.Span(),
			"ceir: tabulate lambda must have exactly 1 parameter, got %d", len(lambda.Params))
	}

	loopCtx := &lowerCtx{
		nextID:       c.nextID,
		bindings:     make(map[string]Value),
		fnSigs:       c.fnSigs,
		monoVersions: c.monoVersions,
		shapeSubst:   c.shapeSubst,
	}
	for k, v := range c.bindings {
		loopCtx.bindings[k] = v
	}

	idx := loopCtx.fresh(types.I32())
	loopCtx.bindings[lambda.Params[0].Name] = idx

	elemVal, err := loopCtx.lowerExpr(lambda.Body)
	if err != nil {
		return Value{}, err
	}
	c.nextID = loopCtx.nextID

	rt := elemVal.Type.WithShape([]types.ShapeDim{types.ConcreteDim(count)})
	res := c.fresh(rt)
	c.insts = append(c.insts, Inst{
		Result:    res,
		Op:        OpTabulateLoop,
		Imm:       count,
		BodyInsts: loopCtx.insts,
		LoopInID:  idx.ID,
		LoopOutID: elemVal.ID,
	})
	return res, nil
}

// lowerScan lowers scan(init, |a, x| f(a, x), v).
// For small counts (<= 32) the lambda body is unrolled and every
// per-iteration accumulator value is collected into an OpTensorLit.
// For larger counts, OpScanLoop is emitted so MIR can lower it to a
// counted runtime loop that walks the input tensor element-by-element.
func (c *lowerCtx) lowerScan(n *ast.Scan) (Value, error) {
	v, err := c.lowerExpr(n.V)
	if err != nil {
		return Value{}, err
	}
	if !v.Type.IsTensor() || v.Type.Rank() != 1 {
		return Value{}, diag.Errorf(n.V.Span(), "ceir: scan requires rank-1 tensor, got %s", v.Type)
	}
	count := v.Type.Shape[0].Val
	if count <= 0 {
		return Value{}, diag.Errorf(n.V.Span(), "ceir: scan requires concrete tensor size")
	}
	elemType := v.Type.ElemType()

	acc, err := c.lowerExpr(n.Init)
	if err != nil {
		return Value{}, err
	}

	const unrollThreshold = 32
	if count <= unrollThreshold {
		outputs := make([]Value, 0, count)
		for i := int64(0); i < count; i++ {
			elem := c.fresh(elemType)
			c.insts = append(c.insts, Inst{Result: elem, Op: OpTensorElem, Args: []Value{v}, Imm: i})

			newAcc, err := c.applyBinaryFn(n.Fn, acc, elem)
			if err != nil {
				return Value{}, err
			}
			outputs = append(outputs, newAcc)
			acc = newAcc
		}
		rt := elemType.WithShape([]types.ShapeDim{types.ConcreteDim(count)})
		res := c.fresh(rt)
		c.insts = append(c.insts, Inst{Result: res, Op: OpTensorLit, Args: outputs})
		return res, nil
	}

	// Large count: lower the binary lambda body once with two fresh
	// loop-input values (acc, elem) and emit OpScanLoop.
	lambda, ok := n.Fn.(*ast.Lambda)
	if !ok {
		return Value{}, diag.Errorf(n.Fn.Span(),
			"ceir: large-N scan (count %d) requires a lambda function", count)
	}
	if len(lambda.Params) != 2 {
		return Value{}, diag.Errorf(lambda.Span(),
			"ceir: scan lambda must have exactly 2 parameters, got %d", len(lambda.Params))
	}

	loopCtx := &lowerCtx{
		nextID:       c.nextID,
		bindings:     make(map[string]Value),
		fnSigs:       c.fnSigs,
		monoVersions: c.monoVersions,
		shapeSubst:   c.shapeSubst,
	}
	for k, v := range c.bindings {
		loopCtx.bindings[k] = v
	}

	loopAcc := loopCtx.fresh(elemType)
	loopElem := loopCtx.fresh(elemType)
	loopCtx.bindings[lambda.Params[0].Name] = loopAcc
	loopCtx.bindings[lambda.Params[1].Name] = loopElem

	newAcc, err := loopCtx.lowerExpr(lambda.Body)
	if err != nil {
		return Value{}, err
	}
	c.nextID = loopCtx.nextID

	rt := elemType.WithShape([]types.ShapeDim{types.ConcreteDim(count)})
	res := c.fresh(rt)
	c.insts = append(c.insts, Inst{
		Result:    res,
		Op:        OpScanLoop,
		Imm:       count,
		Args:      []Value{acc, v},
		BodyInsts: loopCtx.insts,
		LoopInID:  loopAcc.ID,
		LoopInID2: loopElem.ID,
		LoopOutID: newAcc.ID,
	})
	return res, nil
}

// lowerIterateUntil unrolls iterate_until(init, step, pred, max) into a
// chain that runs all max iterations but freezes the state once pred
// has been satisfied via OpSelect cascade.
func (c *lowerCtx) lowerIterateUntil(n *ast.IterateUntil) (Value, error) {
	maxN, err := types.EvalConstInt(n.Max)
	if err != nil {
		return Value{}, err
	}
	if maxN <= 0 {
		return Value{}, diag.Errorf(n.Max.Span(), "ceir: iterate_until requires positive max, got %d", maxN)
	}

	state, err := c.lowerExpr(n.Init)
	if err != nil {
		return Value{}, err
	}

	// pDone starts false.
	pDone := c.fresh(types.Bool())
	c.insts = append(c.insts, Inst{Result: pDone, Op: OpConstBool, ImmB: false})

	for i := int64(0); i < maxN; i++ {
		// Step the state.
		sStep, err := c.applyUnaryFn(n.Step, state)
		if err != nil {
			return Value{}, err
		}
		// Freeze if pDone was already set: state = select(pDone, state, sStep).
		newState := c.fresh(state.Type)
		c.insts = append(c.insts, Inst{Result: newState, Op: OpSelect, Args: []Value{pDone, state, sStep}})

		// Check predicate on the (possibly frozen) state.
		p, err := c.applyUnaryFn(n.Pred, newState)
		if err != nil {
			return Value{}, err
		}

		// pDone = pDone || p
		newPDone := c.fresh(types.Bool())
		c.insts = append(c.insts, Inst{Result: newPDone, Op: OpOr, Args: []Value{pDone, p}})

		state = newState
		pDone = newPDone
	}
	return state, nil
}

// lowerEach unrolls each(v, fn) into one OpTensorElem + OpCall per
// element. fn must already be validated by the type checker as a
// callable identifier whose effects are a subset of the enclosing
// function's effects (v0.13).
func (c *lowerCtx) lowerEach(n *ast.Each) (Value, error) {
	v, err := c.lowerExpr(n.V)
	if err != nil {
		return Value{}, err
	}
	if !v.Type.IsTensor() || v.Type.Rank() != 1 {
		return Value{}, diag.Errorf(n.V.Span(), "ceir: each requires rank-1 tensor, got %s", v.Type)
	}
	count := v.Type.Shape[0].Val
	if count <= 0 {
		return Value{}, diag.Errorf(n.V.Span(), "ceir: each requires concrete tensor size")
	}
	elemType := v.Type.ElemType()

	ident, ok := n.Fn.(*ast.Ident)
	if !ok {
		return Value{}, diag.Errorf(n.Fn.Span(), "ceir: each fn must be a direct identifier")
	}
	sig, ok := c.fnSigs[ident.Name]
	if !ok {
		return Value{}, diag.Errorf(ident.Span(), "ceir: unknown function %q", ident.Name)
	}

	const unrollThreshold = 32
	if count <= unrollThreshold {
		for i := int64(0); i < count; i++ {
			elem := c.fresh(elemType)
			c.insts = append(c.insts, Inst{Result: elem, Op: OpTensorElem, Args: []Value{v}, Imm: i})

			callRes := c.fresh(sig.RetType)
			c.insts = append(c.insts, Inst{Result: callRes, Op: OpCall, FnName: ident.Name, Args: []Value{elem}})
		}
		// each returns unit.
		return c.fresh(types.Unit()), nil
	}

	// Large count: emit OpEachLoop so MIR can lower it to a counted
	// runtime loop. The body is just the per-element call to the
	// callee identified by FnName.
	_ = sig
	res := c.fresh(types.Unit())
	c.insts = append(c.insts, Inst{
		Result: res,
		Op:     OpEachLoop,
		Imm:    count,
		Args:   []Value{v},
		FnName: ident.Name,
	})
	return res, nil
}

func (c *lowerCtx) lowerStmt(s ast.Stmt) error {
	switch n := s.(type) {
	case *ast.LetStmt:
		v, err := c.lowerExpr(n.Init)
		if err != nil {
			return err
		}
		c.bindings[n.Name] = v
		return nil

	case *ast.ExprStmt:
		_, err := c.lowerExpr(n.X)
		return err

	case *ast.ReturnStmt:
		// Return is handled by the function's result; this is a no-op in v0.1
		// (we don't have early returns yet, just single-exit functions)
		return nil
	}
	return diag.Errorf(s.Span(), "ceir: unsupported statement %T", s)
}

func binOpToOp(op string, lType, rType *types.Type) (Op, *types.Type) {
	// Check if this is a tensor operation
	isTensor := lType.IsTensor()

	switch op {
	case "+":
		return OpAdd, lType
	case ".+":
		if isTensor {
			return OpTensorAdd, lType
		}
		return OpAdd, lType
	case "-":
		return OpSub, lType
	case ".-":
		if isTensor {
			return OpTensorSub, lType
		}
		return OpSub, lType
	case "*":
		return OpMul, lType
	case "@":
		// Matrix multiply: f32[M,K] @ f32[K,N] → f32[M,N]
		M := lType.Shape[0]
		N := rType.Shape[1]
		resultType := lType.ElemType().WithShape([]types.ShapeDim{M, N})
		return OpMatMul, resultType
	case ".*":
		if isTensor {
			return OpTensorMul, lType
		}
		return OpMul, lType
	case "/":
		return OpDiv, lType
	case "./":
		if isTensor {
			return OpTensorDiv, lType
		}
		return OpDiv, lType
	case "%", ".%":
		return OpMod, lType
	case "==":
		return OpEq, types.Bool()
	case "!=":
		return OpNe, types.Bool()
	case "<":
		return OpLt, types.Bool()
	case "<=":
		return OpLe, types.Bool()
	case ">":
		return OpGt, types.Bool()
	case ">=":
		return OpGe, types.Bool()
	case "&&":
		return OpAnd, types.Bool()
	case "||":
		return OpOr, types.Bool()
	}
	return OpInvalid, lType
}
