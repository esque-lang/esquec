package types

import (
	"fmt"
	"sort"
	"strings"

	"github.com/esque-lang/esquec/internal/ast"
	"github.com/esque-lang/esquec/internal/diag"
)

// CheckedFile is the typed form of an ast.File.
type CheckedFile struct {
	Path string
	Fns  []*CheckedFn
}

// CheckedFn is a typed function declaration.
type CheckedFn struct {
	Name        string
	ShapeParams []string // shape parameter names (empty for monomorphized)
	Params      []CheckedParam
	RetType     *Type
	Effects     EffectSet // declared effect set (e.g. @io)
	Body        *Type     // type of the body expression
	AST         *ast.FnDecl
	// For monomorphized functions:
	IsMonomorphized bool
	OriginalName    string           // original generic function name
	ShapeValues     map[string]int64 // shape param -> concrete value
}

// CheckedParam is a typed parameter.
type CheckedParam struct {
	Name string
	Type *Type
}

// FnSig is a function signature for call resolution.
type FnSig struct {
	ShapeParams []string // shape parameter names
	Params      []*Type
	RetType     *Type
	Effects     EffectSet // declared effects (e.g. @io); empty means pure
}

// GenericFnDecl stores info about a generic function for monomorphization.
type GenericFnDecl struct {
	AST         *ast.FnDecl
	ShapeParams []string
}

// Instantiation records a specific instantiation of a generic function.
type Instantiation struct {
	FnName      string
	ShapeValues map[string]int64
}

// InstKey returns a unique string key for an instantiation.
func (inst *Instantiation) InstKey() string {
	if len(inst.ShapeValues) == 0 {
		return inst.FnName
	}
	// Sort keys for consistent naming
	var keys []string
	for k := range inst.ShapeValues {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%d", inst.ShapeValues[k]))
	}
	// Use __ as separator to avoid issues with $ in symbol names
	return fmt.Sprintf("%s__%s", inst.FnName, strings.Join(parts, "_"))
}

// Check type-checks a parsed file.
func Check(f *ast.File) (*CheckedFile, error) {
	checker := &fileChecker{
		fnSigs:        make(map[string]*FnSig),
		genericFns:    make(map[string]*GenericFnDecl),
		instantiations: make(map[string]*Instantiation),
	}

	// Pre-register runtime intrinsics so user code can call them. The
	// backend synthesizes their bodies on demand; user-defined functions
	// in the source file will overwrite these entries (the backend
	// detects the collision and skips synthesis in that case).
	registerRuntimeBuiltins(checker.fnSigs)

	// First pass: collect function signatures
	for _, it := range f.Items {
		switch n := it.(type) {
		case *ast.FnDecl:
			var shapeParams []string
			for _, sp := range n.ShapeParams {
				shapeParams = append(shapeParams, sp.Name)
			}

			var params []*Type
			for _, p := range n.Params {
				t, err := resolveType(p.Type)
				if err != nil {
					return nil, err
				}
				params = append(params, t)
			}
			rt, err := resolveType(n.RetType)
			if err != nil {
				return nil, err
			}
			eff, err := effectsFromAttrs(n.Attrs)
			if err != nil {
				return nil, err
			}
			checker.fnSigs[n.Name] = &FnSig{
				ShapeParams: shapeParams,
				Params:      params,
				RetType:     rt,
				Effects:     eff,
			}

			// Track generic functions for monomorphization
			if len(shapeParams) > 0 {
				checker.genericFns[n.Name] = &GenericFnDecl{
					AST:         n,
					ShapeParams: shapeParams,
				}
			}
		}
	}

	// Second pass: type check function bodies and collect instantiations
	out := &CheckedFile{Path: f.Path}
	for _, it := range f.Items {
		switch n := it.(type) {
		case *ast.FnDecl:
			// Skip generic functions in initial pass - they'll be monomorphized
			if len(n.ShapeParams) > 0 {
				continue
			}
			cf, err := checker.checkFn(n)
			if err != nil {
				return nil, err
			}
			out.Fns = append(out.Fns, cf)
		default:
			return nil, diag.Errorf(it.Span(), "unsupported top-level item")
		}
	}

	// Third pass: monomorphize generic functions
	monoFns, err := checker.monomorphize()
	if err != nil {
		return nil, err
	}
	out.Fns = append(out.Fns, monoFns...)

	return out, nil
}

type fileChecker struct {
	fnSigs         map[string]*FnSig
	genericFns     map[string]*GenericFnDecl
	instantiations map[string]*Instantiation // key -> instantiation
}

// registerRuntimeBuiltins seeds fnSigs with the signatures of runtime
// intrinsics. Their machine-code bodies are emitted by the backend's
// runtime helpers, not produced from the user's AST. All print
// intrinsics carry the `@io` effect — callers must therefore also be
// `@io`.
func registerRuntimeBuiltins(fnSigs map[string]*FnSig) {
	fnSigs["print_i32"] = &FnSig{
		Params:  []*Type{I32()},
		RetType: I32(),
		Effects: EffIO,
	}
	fnSigs["print_f32"] = &FnSig{
		Params:  []*Type{F32()},
		RetType: F32(),
		Effects: EffIO,
	}
	// print_str takes the canonical fat-pointer string and writes its
	// bytes to stdout. The pseudo "return value" is the same string so
	// that print_str can sit in expression position; v0.12 has no path
	// that observes the result.
	fnSigs["print_str"] = &FnSig{
		Params:  []*Type{String()},
		RetType: String(),
		Effects: EffIO,
	}
}

// effectsFromAttrs distils an Attr list into an EffectSet. Today only
// `@io` is recognised as an effect annotation; other attributes
// (`@kernel`, `@grad`, `@inline`) are passed through to their own
// passes and ignored here. Unknown `@`-attributes are not rejected
// here — the parser already validates the syntactic form, and a
// future linter can warn on unknown names.
func effectsFromAttrs(attrs []ast.Attr) (EffectSet, error) {
	var e EffectSet
	for _, a := range attrs {
		switch a.Name {
		case "io":
			if len(a.Args) != 0 {
				return 0, diag.Errorf(a.SpanRng, "@io takes no arguments")
			}
			e = e.Union(EffIO)
		}
	}
	return e, nil
}

func (fc *fileChecker) checkFn(f *ast.FnDecl) (*CheckedFn, error) {
	env := map[string]*Type{}

	var ps []CheckedParam
	for _, p := range f.Params {
		t, err := resolveType(p.Type)
		if err != nil {
			return nil, err
		}
		if t.K == KString {
			return nil, diag.Errorf(p.Type.Span(),
				"v0.12 functions cannot have string parameters; only the "+
					"runtime intrinsic print_str accepts a string. "+
					"See docs/reference/planned/strings.md.")
		}
		env[p.Name] = t
		ps = append(ps, CheckedParam{Name: p.Name, Type: t})
	}
	rt, err := resolveType(f.RetType)
	if err != nil {
		return nil, err
	}
	if rt.K == KString {
		return nil, diag.Errorf(f.RetType.Span(),
			"v0.12 functions cannot return string; multi-register fat-"+
				"pointer returns are planned for a later milestone. "+
				"See docs/reference/planned/strings.md.")
	}
	eff, err := effectsFromAttrs(f.Attrs)
	if err != nil {
		return nil, err
	}
	ctx := &checkCtx{
		env:         env,
		fnSigs:      fc.fnSigs,
		retType:     rt,
		fnName:      f.Name,
		fnEffects:   eff,
		fileChecker: fc,
	}
	bt, err := ctx.checkExpr(f.Body)
	if err != nil {
		return nil, err
	}
	if !bt.Equal(rt) {
		return nil, diag.Errorf(f.Body.Span(),
			"function %q: body type %s does not match return type %s",
			f.Name, bt, rt)
	}
	return &CheckedFn{
		Name:    f.Name,
		Params:  ps,
		RetType: rt,
		Effects: eff,
		Body:    bt,
		AST:     f,
	}, nil
}

// checkFnWithSubst checks a function with shape substitutions applied.
func (fc *fileChecker) checkFnWithSubst(f *ast.FnDecl, subst map[string]int64) (*CheckedFn, error) {
	env := map[string]*Type{}

	var ps []CheckedParam
	for _, p := range f.Params {
		t, err := resolveTypeWithSubst(p.Type, subst)
		if err != nil {
			return nil, err
		}
		env[p.Name] = t
		ps = append(ps, CheckedParam{Name: p.Name, Type: t})
	}
	rt, err := resolveTypeWithSubst(f.RetType, subst)
	if err != nil {
		return nil, err
	}

	// Create specialized fnSigs for this instantiation
	specializedSigs := make(map[string]*FnSig)
	for name, sig := range fc.fnSigs {
		specializedSigs[name] = sig
	}

	eff, err := effectsFromAttrs(f.Attrs)
	if err != nil {
		return nil, err
	}
	ctx := &checkCtx{
		env:         env,
		fnSigs:      specializedSigs,
		retType:     rt,
		fnName:      f.Name,
		fnEffects:   eff,
		fileChecker: fc,
		shapeSubst:  subst,
	}
	bt, err := ctx.checkExpr(f.Body)
	if err != nil {
		return nil, err
	}
	if !bt.Equal(rt) {
		return nil, diag.Errorf(f.Body.Span(),
			"function %q: body type %s does not match return type %s",
			f.Name, bt, rt)
	}

	inst := &Instantiation{FnName: f.Name, ShapeValues: subst}
	return &CheckedFn{
		Name:            inst.InstKey(),
		Params:          ps,
		RetType:         rt,
		Effects:         eff,
		Body:            bt,
		AST:             f,
		IsMonomorphized: true,
		OriginalName:    f.Name,
		ShapeValues:     subst,
	}, nil
}

// monomorphize creates specialized versions of generic functions.
func (fc *fileChecker) monomorphize() ([]*CheckedFn, error) {
	var result []*CheckedFn

	// Process instantiations in sorted order for determinism
	var keys []string
	for k := range fc.instantiations {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		inst := fc.instantiations[key]
		gfn, ok := fc.genericFns[inst.FnName]
		if !ok {
			continue
		}

		cf, err := fc.checkFnWithSubst(gfn.AST, inst.ShapeValues)
		if err != nil {
			return nil, err
		}
		result = append(result, cf)
	}

	return result, nil
}

type checkCtx struct {
	env         map[string]*Type
	fnSigs      map[string]*FnSig
	retType     *Type
	fnName      string    // name of the function being checked (for diagnostics)
	fnEffects   EffectSet // declared effects of the enclosing fn
	fileChecker *fileChecker
	shapeSubst  map[string]int64 // current shape substitutions
}

func (c *checkCtx) clone() *checkCtx {
	env2 := make(map[string]*Type, len(c.env))
	for k, v := range c.env {
		env2[k] = v
	}
	return &checkCtx{
		env:         env2,
		fnSigs:      c.fnSigs,
		retType:     c.retType,
		fnName:      c.fnName,
		fnEffects:   c.fnEffects,
		fileChecker: c.fileChecker,
		shapeSubst:  c.shapeSubst,
	}
}

func resolveType(t ast.TypeExpr) (*Type, error) {
	return resolveTypeWithSubst(t, nil)
}

func resolveTypeWithSubst(t ast.TypeExpr, subst map[string]int64) (*Type, error) {
	switch n := t.(type) {
	case *ast.NamedType:
		ty := Lookup(n.Name)
		if ty == nil {
			return nil, diag.Errorf(n.Span(), "unknown type %q", n.Name)
		}
		return ty, nil

	case *ast.TensorType:
		// Resolve element type
		elemTy, err := resolveTypeWithSubst(n.Elem, subst)
		if err != nil {
			return nil, err
		}
		// Resolve shape dimensions
		shape := make([]ShapeDim, len(n.Shape))
		for i, dim := range n.Shape {
			sd, err := resolveShapeDimWithSubst(dim, subst)
			if err != nil {
				return nil, err
			}
			shape[i] = sd
		}
		return elemTy.WithShape(shape), nil
	}
	return nil, diag.Errorf(t.Span(), "unsupported type expression")
}

func resolveShapeDim(s ast.ShapeExpr) (ShapeDim, error) {
	return resolveShapeDimWithSubst(s, nil)
}

// EvalConstInt folds an integer expression AST to a concrete int64
// at type-check time. It supports IntLit, the four arithmetic BinOp
// forms (`+`, `-`, `*`, `/`, `%`), and unary `-`. Identifier lookups
// are intentionally unsupported in v0.11 — there is no `const`
// keyword yet, and `let` bindings are runtime values. Anything
// outside the constant-foldable subset returns an error pointing at
// the offending sub-expression so the caller can wrap it with a
// context-specific message.
//
// Exported because CEIR lowering re-runs the same evaluation on a
// few shape-arithmetic positions (range bounds, tabulate N,
// iterate_until max) when materialising the value into MIR.
func EvalConstInt(e ast.Expr) (int64, error) {
	switch n := e.(type) {
	case *ast.IntLit:
		return n.Value, nil
	case *ast.Unary:
		v, err := EvalConstInt(n.X)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case "-":
			return -v, nil
		}
		return 0, diag.Errorf(n.Span(),
			"unary %q is not a constant integer operation", n.Op)
	case *ast.BinOp:
		l, err := EvalConstInt(n.L)
		if err != nil {
			return 0, err
		}
		r, err := EvalConstInt(n.R)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case "+":
			return l + r, nil
		case "-":
			return l - r, nil
		case "*":
			return l * r, nil
		case "/":
			if r == 0 {
				return 0, diag.Errorf(n.Span(), "division by zero in constant expression")
			}
			return l / r, nil
		case "%":
			if r == 0 {
				return 0, diag.Errorf(n.Span(), "modulo by zero in constant expression")
			}
			return l % r, nil
		}
		return 0, diag.Errorf(n.Span(),
			"binary %q is not a constant integer operation", n.Op)
	}
	return 0, diag.Errorf(e.Span(),
		"expected a constant integer expression")
}

func resolveShapeDimWithSubst(s ast.ShapeExpr, subst map[string]int64) (ShapeDim, error) {
	switch n := s.(type) {
	case *ast.ShapeLit:
		return ConcreteDim(n.Value), nil
	case *ast.ShapeVar:
		// Check if we have a substitution
		if subst != nil {
			if v, ok := subst[n.Name]; ok {
				return ConcreteDim(v), nil
			}
		}
		return VarDim(n.Name), nil
	case *ast.ShapeBinOp:
		// Try to evaluate if both sides are concrete
		left, err := resolveShapeDimWithSubst(n.L, subst)
		if err != nil {
			return ShapeDim{}, err
		}
		right, err := resolveShapeDimWithSubst(n.R, subst)
		if err != nil {
			return ShapeDim{}, err
		}
		if left.Var == "" && right.Var == "" {
			// Both concrete, evaluate
			switch n.Op {
			case "+":
				return ConcreteDim(left.Val + right.Val), nil
			case "-":
				return ConcreteDim(left.Val - right.Val), nil
			case "*":
				return ConcreteDim(left.Val * right.Val), nil
			case "/":
				if right.Val == 0 {
					return ShapeDim{}, diag.Errorf(s.Span(), "division by zero in shape expression")
				}
				return ConcreteDim(left.Val / right.Val), nil
			}
		}
		// Not fully concrete, represent as variable
		return VarDim(shapeExprToString(s)), nil
	}
	return ShapeDim{}, diag.Errorf(s.Span(), "unsupported shape expression")
}

func shapeExprToString(s ast.ShapeExpr) string {
	switch n := s.(type) {
	case *ast.ShapeLit:
		return fmt.Sprintf("%d", n.Value)
	case *ast.ShapeVar:
		return n.Name
	case *ast.ShapeBinOp:
		return fmt.Sprintf("(%s %s %s)", shapeExprToString(n.L), n.Op, shapeExprToString(n.R))
	}
	return "?"
}

func (c *checkCtx) checkExpr(e ast.Expr) (*Type, error) {
	switch n := e.(type) {
	case *ast.IntLit:
		return c.checkIntLit(n)

	case *ast.FloatLit:
		if n.Width == 64 {
			return F64(), nil
		}
		return F32(), nil

	case *ast.BoolLit:
		return Bool(), nil

	case *ast.StringLit:
		return String(), nil

	case *ast.Ident:
		if t, ok := c.env[n.Name]; ok {
			return t, nil
		}
		return nil, diag.Errorf(n.Span(), "unknown identifier %q", n.Name)

	case *ast.BinOp:
		lt, err := c.checkExpr(n.L)
		if err != nil {
			return nil, err
		}
		rt, err := c.checkExpr(n.R)
		if err != nil {
			return nil, err
		}
		switch n.Op {
		case "+", "-", "*", "/", "%":
			// Scalar arithmetic
			if !lt.IsNumeric() {
				return nil, diag.Errorf(n.L.Span(), "operator %q requires numeric type, got %s", n.Op, lt)
			}
			if !lt.Equal(rt) {
				return nil, diag.Errorf(n.Span(), "operator %q: operand types %s and %s differ", n.Op, lt, rt)
			}
			return lt, nil
		case ".+", ".-", ".*", "./", ".%":
			// Element-wise tensor operations
			if !lt.IsNumeric() {
				return nil, diag.Errorf(n.L.Span(), "operator %q requires numeric type, got %s", n.Op, lt)
			}
			if !lt.Equal(rt) {
				return nil, diag.Errorf(n.Span(), "operator %q: operand types %s and %s differ", n.Op, lt, rt)
			}
			// For tensors, ensure shapes match
			if lt.IsTensor() && rt.IsTensor() {
				if len(lt.Shape) != len(rt.Shape) {
					return nil, diag.Errorf(n.Span(), "operator %q: shapes %s and %s have different ranks", n.Op, lt, rt)
				}
				for i := range lt.Shape {
					if !lt.Shape[i].Equal(rt.Shape[i]) {
						return nil, diag.Errorf(n.Span(), "operator %q: shapes %s and %s differ at dimension %d", n.Op, lt, rt, i)
					}
				}
			}
			return lt, nil
		case "@":
			// Matrix multiplication: f32[M,K] @ f32[K,N] → f32[M,N]
			if !lt.IsNumeric() {
				return nil, diag.Errorf(n.L.Span(), "operator %q requires numeric type, got %s", n.Op, lt)
			}
			if !lt.IsTensor() || lt.Rank() != 2 {
				return nil, diag.Errorf(n.L.Span(), "operator @ requires rank-2 left operand, got %s", lt)
			}
			if !rt.IsTensor() || rt.Rank() != 2 {
				return nil, diag.Errorf(n.R.Span(), "operator @ requires rank-2 right operand, got %s", rt)
			}
			// Element types must match
			if lt.K != rt.K {
				return nil, diag.Errorf(n.Span(), "operator @: element types %s and %s differ", lt.ElemType(), rt.ElemType())
			}
			// Shape check: [M,K] @ [K,N] → [M,N]
			lK := lt.Shape[1]
			rK := rt.Shape[0]
			if !lK.Equal(rK) {
				return nil, diag.Errorf(n.Span(), "matmul inner dimensions mismatch: %v vs %v", lK, rK)
			}
			M := lt.Shape[0]
			N := rt.Shape[1]
			return lt.ElemType().WithShape([]ShapeDim{M, N}), nil
		case "==", "!=", "<", "<=", ">", ">=":
			if !lt.Equal(rt) {
				return nil, diag.Errorf(n.Span(), "comparison %q: operand types %s and %s differ", n.Op, lt, rt)
			}
			if lt.K == KString {
				return nil, diag.Errorf(n.Span(),
					"comparing strings is not supported in v0.12; v0.12 strings "+
						"are immutable fat pointers consumable only by print_str. "+
						"See docs/reference/planned/strings.md.")
			}
			return Bool(), nil
		case "&&", "||":
			if !lt.Equal(Bool()) || !rt.Equal(Bool()) {
				return nil, diag.Errorf(n.Span(), "operator %q requires bool operands", n.Op)
			}
			return Bool(), nil
		}
		return nil, diag.Errorf(n.Span(), "unsupported binary operator %q", n.Op)

	case *ast.Pipeline:
		// x |> f desugars to f(x)
		lt, err := c.checkExpr(n.L)
		if err != nil {
			return nil, err
		}
		switch rhs := n.R.(type) {
		case *ast.Ident:
			sig, ok := c.fnSigs[rhs.Name]
			if !ok {
				return nil, diag.Errorf(rhs.Span(), "unknown function %q", rhs.Name)
			}
			// Check for generic function
			if len(sig.ShapeParams) > 0 {
				// Infer shape arguments from the piped value
				shapeArgs, err := c.inferShapeArgs(sig, []*Type{lt}, nil, n.Span())
				if err != nil {
					return nil, err
				}
				return c.checkGenericCall(rhs.Name, sig, shapeArgs, []*Type{lt}, n.Span())
			}
			if len(sig.Params) != 1 {
				return nil, diag.Errorf(n.Span(), "pipeline function %q expects 1 param, has %d",
					rhs.Name, len(sig.Params))
			}
			if !lt.Equal(sig.Params[0]) {
				return nil, diag.Errorf(n.L.Span(), "pipeline: expected %s, got %s",
					sig.Params[0], lt)
			}
			return sig.RetType, nil
		case *ast.Call:
			ident, ok := rhs.Fn.(*ast.Ident)
			if !ok {
				return nil, diag.Errorf(rhs.Fn.Span(), "pipeline callee must be an identifier")
			}
			sig, ok := c.fnSigs[ident.Name]
			if !ok {
				return nil, diag.Errorf(ident.Span(), "unknown function %q", ident.Name)
			}
			// Type check remaining args
			argTypes := []*Type{lt}
			for _, arg := range rhs.Args {
				at, err := c.checkExpr(arg)
				if err != nil {
					return nil, err
				}
				argTypes = append(argTypes, at)
			}
			// Check for generic function
			if len(sig.ShapeParams) > 0 {
				shapeArgs, err := c.inferShapeArgs(sig, argTypes, rhs.ShapeArgs, n.Span())
				if err != nil {
					return nil, err
				}
				return c.checkGenericCall(ident.Name, sig, shapeArgs, argTypes, n.Span())
			}
			if len(argTypes) != len(sig.Params) {
				return nil, diag.Errorf(n.Span(), "function %q expects %d args, got %d",
					ident.Name, len(sig.Params), len(argTypes))
			}
			for i, at := range argTypes {
				if !at.Equal(sig.Params[i]) {
					return nil, diag.Errorf(n.Span(), "argument %d: expected %s, got %s",
						i+1, sig.Params[i], at)
				}
			}
			return sig.RetType, nil
		default:
			return nil, diag.Errorf(n.R.Span(), "pipeline right side must be function or call")
		}

	case *ast.Reduce:
		xt, err := c.checkExpr(n.X)
		if err != nil {
			return nil, err
		}
		if !xt.IsNumeric() {
			return nil, diag.Errorf(n.X.Span(), "reduction requires numeric type, got %s", xt)
		}
		if xt.IsTensor() {
			if xt.Rank() == 1 {
				return xt.ElemType(), nil
			}
			newShape := make([]ShapeDim, xt.Rank()-1)
			copy(newShape, xt.Shape[:xt.Rank()-1])
			return xt.ElemType().WithShape(newShape), nil
		}
		return xt, nil

	case *ast.TensorLit:
		if len(n.Elems) == 0 {
			return nil, diag.Errorf(n.Span(), "empty tensor literal")
		}
		var elemType *Type
		for i, elem := range n.Elems {
			et, err := c.checkExpr(elem)
			if err != nil {
				return nil, err
			}
			if i == 0 {
				elemType = et
			} else if !et.Equal(elemType) {
				return nil, diag.Errorf(elem.Span(), "tensor element %d has type %s, expected %s", i+1, et, elemType)
			}
		}
		if elemType.IsTensor() {
			innerShape := elemType.Shape
			outerDim := ConcreteDim(int64(len(n.Elems)))
			resultShape := append([]ShapeDim{outerDim}, innerShape...)
			return elemType.ElemType().WithShape(resultShape), nil
		}
		shape := []ShapeDim{ConcreteDim(int64(len(n.Elems)))}
		return elemType.WithShape(shape), nil

	case *ast.Unary:
		xt, err := c.checkExpr(n.X)
		if err != nil {
			return nil, err
		}
		switch n.Op {
		case "-":
			if !xt.IsNumeric() {
				return nil, diag.Errorf(n.X.Span(), "negation requires numeric type, got %s", xt)
			}
			return xt, nil
		case "!":
			if !xt.Equal(Bool()) {
				return nil, diag.Errorf(n.X.Span(), "logical not requires bool, got %s", xt)
			}
			return Bool(), nil
		}
		return nil, diag.Errorf(n.Span(), "unsupported unary operator %q", n.Op)

	case *ast.Call:
		ident, ok := n.Fn.(*ast.Ident)
		if !ok {
			return nil, diag.Errorf(n.Fn.Span(), "callee must be an identifier")
		}
		sig, ok := c.fnSigs[ident.Name]
		if !ok {
			return nil, diag.Errorf(ident.Span(), "unknown function %q", ident.Name)
		}
		// Effect propagation: the callee's effects must be a subset of
		// the caller's. A pure function therefore cannot call an `@io`
		// function, but an `@io` function can call any pure helper.
		if !sig.Effects.Subset(c.fnEffects) {
			missing := sig.Effects &^ c.fnEffects
			return nil, diag.Errorf(n.Span(),
				"function %q has effect %s but the enclosing function %q "+
					"is not annotated with that effect; add the annotation "+
					"to the caller, e.g. `@io fn %s(...)`.",
				ident.Name, missing, c.fnName, c.fnName)
		}
		// Type check arguments
		var argTypes []*Type
		for _, arg := range n.Args {
			at, err := c.checkExpr(arg)
			if err != nil {
				return nil, err
			}
			argTypes = append(argTypes, at)
		}
		// Check for generic function
		if len(sig.ShapeParams) > 0 {
			shapeArgs, err := c.inferShapeArgs(sig, argTypes, n.ShapeArgs, n.Span())
			if err != nil {
				return nil, err
			}
			return c.checkGenericCall(ident.Name, sig, shapeArgs, argTypes, n.Span())
		}
		// Non-generic function
		if len(n.Args) != len(sig.Params) {
			return nil, diag.Errorf(n.Span(), "function %q expects %d args, got %d",
				ident.Name, len(sig.Params), len(n.Args))
		}
		for i, at := range argTypes {
			if !at.Equal(sig.Params[i]) {
				return nil, diag.Errorf(n.Args[i].Span(), "argument %d: expected %s, got %s",
					i+1, sig.Params[i], at)
			}
		}
		return sig.RetType, nil

	case *ast.Block:
		inner := c.clone()
		for _, stmt := range n.Stmts {
			if err := inner.checkStmt(stmt); err != nil {
				return nil, err
			}
		}
		if n.Result == nil {
			return Unit(), nil
		}
		return inner.checkExpr(n.Result)

	case *ast.If:
		ct, err := c.checkExpr(n.Cond)
		if err != nil {
			return nil, err
		}
		if !ct.Equal(Bool()) {
			return nil, diag.Errorf(n.Cond.Span(), "if condition must be bool, got %s", ct)
		}
		tt, err := c.checkExpr(n.Then)
		if err != nil {
			return nil, err
		}
		if n.Else == nil {
			if !tt.Equal(Unit()) {
				return nil, diag.Errorf(n.Then.Span(), "if without else must have unit type, got %s", tt)
			}
			return Unit(), nil
		}
		et, err := c.checkExpr(n.Else)
		if err != nil {
			return nil, err
		}
		if !tt.Equal(et) {
			return nil, diag.Errorf(n.Span(), "if branches have different types: %s vs %s", tt, et)
		}
		return tt, nil

	case *ast.Match:
		st, err := c.checkExpr(n.Scrutinee)
		if err != nil {
			return nil, err
		}
		if len(n.Arms) == 0 {
			return nil, diag.Errorf(n.Span(), "match expression must have at least one arm")
		}
		var resultTy *Type
		for i, arm := range n.Arms {
			if err := c.checkPattern(arm.Pattern, st); err != nil {
				return nil, err
			}
			inner := c.clone()
			if bp, ok := arm.Pattern.(*ast.BindPat); ok {
				inner.env[bp.Name] = st
			}
			if arm.Guard != nil {
				gt, err := inner.checkExpr(arm.Guard)
				if err != nil {
					return nil, err
				}
				if !gt.Equal(Bool()) {
					return nil, diag.Errorf(arm.Guard.Span(), "match guard must be bool, got %s", gt)
				}
			}
			bt, err := inner.checkExpr(arm.Body)
			if err != nil {
				return nil, err
			}
			if i == 0 {
				resultTy = bt
			} else if !bt.Equal(resultTy) {
				return nil, diag.Errorf(arm.Body.Span(), "match arm has type %s, expected %s", bt, resultTy)
			}
		}
		return resultTy, nil

	case *ast.Cast:
		xt, err := c.checkExpr(n.X)
		if err != nil {
			return nil, err
		}
		tt, err := resolveType(n.Target)
		if err != nil {
			return nil, err
		}
		if !canCast(xt, tt) {
			return nil, diag.Errorf(n.Span(), "cannot cast %s to %s", xt, tt)
		}
		return tt, nil

	case *ast.Lambda:
		// For now, lambdas require explicit parameter types or are used in context
		// where types can be inferred (like fold). Store lambda for later resolution.
		// Return a placeholder function type - actual typing happens at use site.
		return nil, diag.Errorf(n.Span(), "standalone lambda expressions not yet supported; use in fold or iterate context")

	case *ast.Fold:
		// f/tensor or f/init/tensor
		// Type check the tensor operand first
		xt, err := c.checkExpr(n.X)
		if err != nil {
			return nil, err
		}
		if !xt.IsTensor() {
			return nil, diag.Errorf(n.X.Span(), "fold requires tensor operand, got %s", xt)
		}
		elemTy := xt.ElemType()

		// Check the fold function
		if lambda, ok := n.Fn.(*ast.Lambda); ok {
			// Lambda fold: validate parameters and body
			if len(lambda.Params) != 2 {
				return nil, diag.Errorf(lambda.Span(), "fold lambda must have exactly 2 parameters, got %d", len(lambda.Params))
			}
			// Infer parameter types from element type and validate body
			inner := c.clone()
			inner.env[lambda.Params[0].Name] = elemTy // accumulator
			inner.env[lambda.Params[1].Name] = elemTy // element
			bodyTy, err := inner.checkExpr(lambda.Body)
			if err != nil {
				return nil, err
			}
			// Body should return element type
			if !bodyTy.Equal(elemTy) {
				return nil, diag.Errorf(lambda.Body.Span(), "fold lambda body has type %s, expected %s", bodyTy, elemTy)
			}
		} else if ident, ok := n.Fn.(*ast.Ident); ok {
			// Named function fold: check it exists (or is a built-in like add, mul)
			if ident.Name != "+" && ident.Name != "add" && ident.Name != "*" && ident.Name != "mul" &&
				ident.Name != "max" && ident.Name != "min" {
				if _, ok := c.fnSigs[ident.Name]; !ok {
					return nil, diag.Errorf(ident.Span(), "unknown fold function %q", ident.Name)
				}
			}
		} else {
			return nil, diag.Errorf(n.Fn.Span(), "fold function must be identifier or lambda")
		}

		// For f/tensor (no init), the function must be (T, T) -> T
		// Result is the element type (reducing tensor to scalar for 1D)
		if n.Init == nil {
			// Reduce: f/tensor -> element type
			if xt.Rank() == 1 {
				return elemTy, nil
			}
			// For higher-rank, reduce last dimension
			newShape := make([]ShapeDim, xt.Rank()-1)
			copy(newShape, xt.Shape[:xt.Rank()-1])
			return elemTy.WithShape(newShape), nil
		}

		// f/init/tensor: check init type matches element type
		initTy, err := c.checkExpr(n.Init)
		if err != nil {
			return nil, err
		}
		// For fold with init, the init type determines the result type
		// (more general than reduce - allows accumulator of different type)
		if xt.Rank() == 1 {
			return initTy, nil
		}
		newShape := make([]ShapeDim, xt.Rank()-1)
		copy(newShape, xt.Shape[:xt.Rank()-1])
		return initTy.WithShape(newShape), nil

	case *ast.RangeExpr:
		return c.checkRangeExpr(n)

	case *ast.Tabulate:
		return c.checkTabulate(n)

	case *ast.Scan:
		return c.checkScan(n)

	case *ast.IterateUntil:
		return c.checkIterateUntil(n)

	case *ast.Each:
		return c.checkEach(n)

	case *ast.Iterate:
		// iterate(n, init, f) where f: T -> T
		// Check n is an integer
		nt, err := c.checkExpr(n.N)
		if err != nil {
			return nil, err
		}
		if !nt.IsInteger() {
			return nil, diag.Errorf(n.N.Span(), "iterate count must be integer, got %s", nt)
		}

		// Check init type
		initTy, err := c.checkExpr(n.Init)
		if err != nil {
			return nil, err
		}

		// Check the iterate function
		if lambda, ok := n.Fn.(*ast.Lambda); ok {
			// Lambda iterate: validate parameter and body
			if len(lambda.Params) != 1 {
				return nil, diag.Errorf(lambda.Span(), "iterate lambda must have exactly 1 parameter, got %d", len(lambda.Params))
			}
			// Infer parameter type from init type and validate body
			inner := c.clone()
			inner.env[lambda.Params[0].Name] = initTy
			bodyTy, err := inner.checkExpr(lambda.Body)
			if err != nil {
				return nil, err
			}
			// Body should return same type as init
			if !bodyTy.Equal(initTy) {
				return nil, diag.Errorf(lambda.Body.Span(), "iterate lambda body has type %s, expected %s", bodyTy, initTy)
			}
		} else if ident, ok := n.Fn.(*ast.Ident); ok {
			// Named function iterate: check it exists
			if _, ok := c.fnSigs[ident.Name]; !ok {
				return nil, diag.Errorf(ident.Span(), "unknown iterate function %q", ident.Name)
			}
		} else {
			return nil, diag.Errorf(n.Fn.Span(), "iterate function must be identifier or lambda")
		}

		// Return the init type as the result type
		return initTy, nil
	}
	return nil, diag.Errorf(e.Span(), "unsupported expression %T", e)
}

// inferShapeArgs infers shape arguments from explicit shape args or argument types.
func (c *checkCtx) inferShapeArgs(sig *FnSig, argTypes []*Type, explicitShapeArgs []ast.ShapeExpr, span diag.Span) (map[string]int64, error) {
	result := make(map[string]int64)

	// First, use explicit shape arguments if provided
	if len(explicitShapeArgs) > 0 {
		if len(explicitShapeArgs) != len(sig.ShapeParams) {
			return nil, diag.Errorf(span, "expected %d shape arguments, got %d",
				len(sig.ShapeParams), len(explicitShapeArgs))
		}
		for i, shapeArg := range explicitShapeArgs {
			dim, err := resolveShapeDimWithSubst(shapeArg, c.shapeSubst)
			if err != nil {
				return nil, err
			}
			if dim.Var != "" {
				return nil, diag.Errorf(span, "shape argument must be concrete, got %s", dim.Var)
			}
			result[sig.ShapeParams[i]] = dim.Val
		}
		return result, nil
	}

	// Infer from argument types
	if len(argTypes) != len(sig.Params) {
		return nil, diag.Errorf(span, "expected %d arguments, got %d",
			len(sig.Params), len(argTypes))
	}

	for i, argType := range argTypes {
		paramType := sig.Params[i]
		if err := c.unifyShapes(paramType, argType, result, span); err != nil {
			return nil, err
		}
	}

	// Verify all shape params were inferred
	for _, sp := range sig.ShapeParams {
		if _, ok := result[sp]; !ok {
			return nil, diag.Errorf(span, "could not infer shape parameter %q", sp)
		}
	}

	return result, nil
}

// unifyShapes extracts shape variable bindings by matching a parameter type with an argument type.
func (c *checkCtx) unifyShapes(paramType, argType *Type, bindings map[string]int64, span diag.Span) error {
	// Element types must match
	if paramType.K != argType.K {
		return diag.Errorf(span, "type mismatch: expected %s, got %s", paramType, argType)
	}

	// If param is not a tensor, nothing to unify
	if !paramType.IsTensor() {
		return nil
	}

	// Arg must also be a tensor with same rank
	if !argType.IsTensor() || len(paramType.Shape) != len(argType.Shape) {
		return diag.Errorf(span, "shape mismatch: expected %s, got %s", paramType, argType)
	}

	// Unify each dimension
	for i := range paramType.Shape {
		pDim := paramType.Shape[i]
		aDim := argType.Shape[i]

		if pDim.Var != "" {
			// Parameter has a shape variable
			if aDim.Var != "" {
				return diag.Errorf(span, "cannot infer shape: argument has shape variable %s", aDim.Var)
			}
			// Bind or check consistency
			if existing, ok := bindings[pDim.Var]; ok {
				if existing != aDim.Val {
					return diag.Errorf(span, "shape variable %s has conflicting values: %d vs %d",
						pDim.Var, existing, aDim.Val)
				}
			} else {
				bindings[pDim.Var] = aDim.Val
			}
		} else {
			// Parameter has concrete dimension - must match
			if aDim.Var != "" || pDim.Val != aDim.Val {
				return diag.Errorf(span, "dimension mismatch: expected %v, got %v", pDim, aDim)
			}
		}
	}

	return nil
}

// checkGenericCall type-checks a call to a generic function with inferred shape args.
func (c *checkCtx) checkGenericCall(fnName string, sig *FnSig, shapeArgs map[string]int64, argTypes []*Type, span diag.Span) (*Type, error) {
	// Register instantiation
	inst := &Instantiation{FnName: fnName, ShapeValues: shapeArgs}
	c.fileChecker.instantiations[inst.InstKey()] = inst

	// Substitute shape variables in parameter types
	if len(argTypes) != len(sig.Params) {
		return nil, diag.Errorf(span, "expected %d arguments, got %d",
			len(sig.Params), len(argTypes))
	}

	for i, argType := range argTypes {
		paramType := substituteShapes(sig.Params[i], shapeArgs)
		if !argType.Equal(paramType) {
			return nil, diag.Errorf(span, "argument %d: expected %s, got %s",
				i+1, paramType, argType)
		}
	}

	// Return type with substitutions
	return substituteShapes(sig.RetType, shapeArgs), nil
}

// substituteShapes replaces shape variables with concrete values.
func substituteShapes(t *Type, subst map[string]int64) *Type {
	if !t.IsTensor() {
		return t
	}

	newShape := make([]ShapeDim, len(t.Shape))
	for i, dim := range t.Shape {
		if dim.Var != "" {
			if v, ok := subst[dim.Var]; ok {
				newShape[i] = ConcreteDim(v)
			} else {
				newShape[i] = dim
			}
		} else {
			newShape[i] = dim
		}
	}
	return t.ElemType().WithShape(newShape)
}

func canCast(from, to *Type) bool {
	if from.Equal(to) {
		return true
	}
	// Strings cannot participate in `as` casts in v0.12. Rejecting both
	// directions keeps the rule symmetric and matches the forward-
	// pointing diagnostic on string ops elsewhere.
	if from.K == KString || to.K == KString {
		return false
	}
	if from.IsNumeric() && !from.IsTensor() && to.IsNumeric() && !to.IsTensor() {
		return true
	}
	return false
}

func (c *checkCtx) checkPattern(p ast.Pattern, scrutTy *Type) error {
	switch n := p.(type) {
	case *ast.WildcardPat:
		return nil
	case *ast.BindPat:
		return nil
	case *ast.LitPat:
		var litTy *Type
		switch lit := n.Value.(type) {
		case *ast.IntLit:
			litTy = I32()
			_ = lit
		case *ast.BoolLit:
			litTy = Bool()
		default:
			return diag.Errorf(n.Span(), "unsupported literal pattern %T", n.Value)
		}
		if !litTy.Equal(scrutTy) {
			return diag.Errorf(n.Span(), "pattern type %s does not match scrutinee type %s", litTy, scrutTy)
		}
		return nil
	}
	return diag.Errorf(p.Span(), "unsupported pattern %T", p)
}

func (c *checkCtx) checkStmt(s ast.Stmt) error {
	switch n := s.(type) {
	case *ast.LetStmt:
		it, err := c.checkExpr(n.Init)
		if err != nil {
			return err
		}
		if n.Type != nil {
			dt, err := resolveType(n.Type)
			if err != nil {
				return err
			}
			if !it.Equal(dt) {
				return diag.Errorf(n.Init.Span(), "let %s: initializer type %s does not match declared type %s",
					n.Name, it, dt)
			}
			c.env[n.Name] = dt
		} else {
			c.env[n.Name] = it
		}
		return nil

	case *ast.ExprStmt:
		_, err := c.checkExpr(n.X)
		return err

	case *ast.ReturnStmt:
		if n.Value == nil {
			if !c.retType.Equal(Unit()) {
				return diag.Errorf(n.Span(), "return without value in function returning %s", c.retType)
			}
			return nil
		}
		vt, err := c.checkExpr(n.Value)
		if err != nil {
			return err
		}
		if !vt.Equal(c.retType) {
			return diag.Errorf(n.Value.Span(), "return type %s does not match function return type %s",
				vt, c.retType)
		}
		return nil
	}
	return diag.Errorf(s.Span(), "unsupported statement %T", s)
}

// rangeMaxLen is the v0.10 ceiling on the number of elements in a
// literal-bounded range expression. Larger ranges would cause runaway
// .rodata growth; non-literal bounds and large-N runtime ranges are a
// follow-up.
const rangeMaxLen = 1 << 20

// unrollLimit is the inline-unroll threshold for tabulate/scan/each.
// Counts at or below this limit are unrolled by the CEIR lowerer;
// counts above this limit emit OpTabulateLoop / OpScanLoop /
// OpEachLoop and are lowered to a runtime loop in MIR (v0.11). The
// upper bound is rangeMaxLen.
const unrollLimit = 32

// iterateUntilUnrollLimit is the unroll ceiling for iterate_until.
// Unlike tabulate/scan/each, iterate_until cannot be expressed as a
// counted loop because it observes a runtime predicate to short-circuit
// the residual iterations. Until OpIterateUntilLoop with predicate-
// driven branching lands, max remains capped at this value.
const iterateUntilUnrollLimit = 32

// intSuffixLimits maps each integer type suffix to its inclusive value
// range. Lookup is keyed by the suffix string the parser stored in
// IntLit.TypeSuffix (without the leading `_`).
var intSuffixLimits = map[string]struct {
	min, max int64
	ty       func() *Type
}{
	"i8":  {-1 << 7, 1<<7 - 1, I8},
	"i16": {-1 << 15, 1<<15 - 1, func() *Type { panic("i16 unreachable") }},
	"i32": {-1 << 31, 1<<31 - 1, I32},
	"i64": {-1 << 63, 1<<63 - 1, I64},
	"u8":  {0, 1<<8 - 1, U8},
	"u16": {0, 1<<16 - 1, func() *Type { panic("u16 unreachable") }},
	"u32": {0, 1<<32 - 1, U32},
	"u64": {0, 1<<63 - 1, func() *Type { panic("u64 unreachable") }}, // capped at i64.max for v0.10
}

// v0.12 codegens i8, u8, i32, i64, and u32 on the CPU backend. The
// remaining integer widths are accepted by the parser but rejected here
// with a clean diagnostic so the limitation is one source-traceable
// error rather than a panic deep in lowering.
var supportedIntSuffixes = map[string]bool{
	"i8":  true,
	"u8":  true,
	"i32": true,
	"i64": true,
	"u32": true,
}

// checkIntLit dispatches an IntLit to the type implied by its suffix
// (defaulting to i32 when unsuffixed) and validates the value fits.
func (c *checkCtx) checkIntLit(n *ast.IntLit) (*Type, error) {
	if n.TypeSuffix == "" {
		// Unsuffixed: default i32. Range-check just like an explicit _i32.
		lim := intSuffixLimits["i32"]
		if n.Value < lim.min || n.Value > lim.max {
			return nil, diag.Errorf(n.Span(),
				"integer literal %d does not fit in i32 (range %d..%d); "+
					"add an explicit suffix like _i64",
				n.Value, lim.min, lim.max)
		}
		return I32(), nil
	}
	lim, ok := intSuffixLimits[n.TypeSuffix]
	if !ok {
		return nil, diag.Errorf(n.Span(),
			"unknown integer suffix _%s", n.TypeSuffix)
	}
	if n.Value < lim.min || n.Value > lim.max {
		return nil, diag.Errorf(n.Span(),
			"integer literal %d does not fit in %s (range %d..%d)",
			n.Value, n.TypeSuffix, lim.min, lim.max)
	}
	if !supportedIntSuffixes[n.TypeSuffix] {
		return nil, diag.Errorf(n.Span(),
			"integer suffix _%s is parseable but not yet supported by the v0.12 "+
				"CPU backend; only _i8, _u8, _i32, _i64, and _u32 currently "+
				"codegen. Other widths are planned for a future milestone.",
			n.TypeSuffix)
	}
	return lim.ty(), nil
}

// checkRangeExpr type-checks a literal-bounded integer range
// `lo..hi` / `lo..=hi`. v0.10 only accepts integer-literal bounds and
// produces an i32 rank-1 tensor.
func (c *checkCtx) checkRangeExpr(n *ast.RangeExpr) (*Type, error) {
	lo, err := EvalConstInt(n.Lo)
	if err != nil {
		return nil, err
	}
	hi, err := EvalConstInt(n.Hi)
	if err != nil {
		return nil, err
	}
	count := hi - lo
	if n.Inclusive {
		count++
	}
	if count <= 0 {
		return nil, diag.Errorf(n.Span(),
			"empty or reversed range: %d..%s%d", lo,
			func() string {
				if n.Inclusive {
					return "="
				}
				return ""
			}(), hi)
	}
	if count > rangeMaxLen {
		return nil, diag.Errorf(n.Span(),
			"range too large: %d elements exceeds v0.10 ceiling of %d", count, rangeMaxLen)
	}
	return I32().WithShape([]ShapeDim{ConcreteDim(count)}), nil
}

// checkTabulate handles tabulate(n, fn).
func (c *checkCtx) checkTabulate(n *ast.Tabulate) (*Type, error) {
	count, err := EvalConstInt(n.N)
	if err != nil {
		return nil, err
	}
	if count <= 0 {
		return nil, diag.Errorf(n.N.Span(),
			"tabulate count must be positive, got %d", count)
	}
	if count > rangeMaxLen {
		return nil, diag.Errorf(n.N.Span(),
			"tabulate count %d exceeds the v0.11 ceiling of %d",
			count, rangeMaxLen)
	}
	bodyTy, err := c.checkUnaryFn(n.Fn, I32(), "tabulate")
	if err != nil {
		return nil, err
	}
	return bodyTy.WithShape([]ShapeDim{ConcreteDim(count)}), nil
}

// checkUnaryFn validates a 1-arg lambda/named-fn whose argument has
// type argTy and returns the function's body/return type.
func (c *checkCtx) checkUnaryFn(fn ast.Expr, argTy *Type, ctxName string) (*Type, error) {
	switch fn := fn.(type) {
	case *ast.Lambda:
		if len(fn.Params) != 1 {
			return nil, diag.Errorf(fn.Span(),
				"%s function must take exactly 1 argument, got %d", ctxName, len(fn.Params))
		}
		inner := c.clone()
		inner.env[fn.Params[0].Name] = argTy
		return inner.checkExpr(fn.Body)
	case *ast.Ident:
		sig, ok := c.fnSigs[fn.Name]
		if !ok {
			return nil, diag.Errorf(fn.Span(), "unknown %s function %q", ctxName, fn.Name)
		}
		if len(sig.Params) != 1 {
			return nil, diag.Errorf(fn.Span(),
				"%s function %q must have exactly 1 parameter, has %d", ctxName, fn.Name, len(sig.Params))
		}
		if !sig.Params[0].Equal(argTy) {
			return nil, diag.Errorf(fn.Span(),
				"%s function %q expects argument %s, got %s", ctxName, fn.Name, sig.Params[0], argTy)
		}
		return sig.RetType, nil
	}
	return nil, diag.Errorf(fn.Span(), "%s function must be a lambda or identifier", ctxName)
}

// checkBinaryFn validates a 2-arg lambda/named-fn whose argument types
// are aTy, bTy. Returns the body/return type.
func (c *checkCtx) checkBinaryFn(fn ast.Expr, aTy, bTy *Type, ctxName string) (*Type, error) {
	switch fn := fn.(type) {
	case *ast.Lambda:
		if len(fn.Params) != 2 {
			return nil, diag.Errorf(fn.Span(),
				"%s function must take exactly 2 arguments, got %d", ctxName, len(fn.Params))
		}
		inner := c.clone()
		inner.env[fn.Params[0].Name] = aTy
		inner.env[fn.Params[1].Name] = bTy
		return inner.checkExpr(fn.Body)
	case *ast.Ident:
		sig, ok := c.fnSigs[fn.Name]
		if !ok {
			return nil, diag.Errorf(fn.Span(), "unknown %s function %q", ctxName, fn.Name)
		}
		if len(sig.Params) != 2 {
			return nil, diag.Errorf(fn.Span(),
				"%s function %q must have exactly 2 parameters, has %d", ctxName, fn.Name, len(sig.Params))
		}
		if !sig.Params[0].Equal(aTy) || !sig.Params[1].Equal(bTy) {
			return nil, diag.Errorf(fn.Span(),
				"%s function %q expects (%s, %s), got (%s, %s)",
				ctxName, fn.Name, sig.Params[0], sig.Params[1], aTy, bTy)
		}
		return sig.RetType, nil
	}
	return nil, diag.Errorf(fn.Span(), "%s function must be a lambda or identifier", ctxName)
}

// checkScan handles scan(init, fn, v) -> typeof(v).
func (c *checkCtx) checkScan(n *ast.Scan) (*Type, error) {
	vTy, err := c.checkExpr(n.V)
	if err != nil {
		return nil, err
	}
	if !vTy.IsTensor() || vTy.Rank() != 1 {
		return nil, diag.Errorf(n.V.Span(),
			"scan requires a rank-1 tensor, got %s", vTy)
	}
	if vTy.Shape[0].Var != "" {
		return nil, diag.Errorf(n.V.Span(),
			"scan requires a tensor with a concrete length")
	}
	count := vTy.Shape[0].Val
	if count <= 0 {
		return nil, diag.Errorf(n.V.Span(),
			"scan tensor must have positive length, got %d", count)
	}
	if count > rangeMaxLen {
		return nil, diag.Errorf(n.V.Span(),
			"scan length %d exceeds the v0.11 ceiling of %d",
			count, rangeMaxLen)
	}
	elemTy := vTy.ElemType()
	initTy, err := c.checkExpr(n.Init)
	if err != nil {
		return nil, err
	}
	if !initTy.Equal(elemTy) {
		return nil, diag.Errorf(n.Init.Span(),
			"scan init has type %s, expected %s to match tensor element type", initTy, elemTy)
	}
	bodyTy, err := c.checkBinaryFn(n.Fn, elemTy, elemTy, "scan")
	if err != nil {
		return nil, err
	}
	if !bodyTy.Equal(elemTy) {
		return nil, diag.Errorf(n.Fn.Span(),
			"scan function returns %s, expected %s", bodyTy, elemTy)
	}
	return elemTy.WithShape([]ShapeDim{ConcreteDim(count)}), nil
}

// checkIterateUntil handles iterate_until(init, step, pred, max).
func (c *checkCtx) checkIterateUntil(n *ast.IterateUntil) (*Type, error) {
	initTy, err := c.checkExpr(n.Init)
	if err != nil {
		return nil, err
	}
	if initTy.IsTensor() {
		return nil, diag.Errorf(n.Init.Span(),
			"iterate_until: tensor state is not supported in v0.10; use a scalar state")
	}
	if !initTy.IsNumeric() && !initTy.Equal(Bool()) {
		return nil, diag.Errorf(n.Init.Span(),
			"iterate_until: init must be a scalar numeric or bool, got %s", initTy)
	}
	stepRet, err := c.checkUnaryFn(n.Step, initTy, "iterate_until step")
	if err != nil {
		return nil, err
	}
	if !stepRet.Equal(initTy) {
		return nil, diag.Errorf(n.Step.Span(),
			"iterate_until step returns %s, expected %s", stepRet, initTy)
	}
	predRet, err := c.checkUnaryFn(n.Pred, initTy, "iterate_until pred")
	if err != nil {
		return nil, err
	}
	if !predRet.Equal(Bool()) {
		return nil, diag.Errorf(n.Pred.Span(),
			"iterate_until pred returns %s, expected bool", predRet)
	}
	maxVal, err := EvalConstInt(n.Max)
	if err != nil {
		return nil, err
	}
	if maxVal <= 0 {
		return nil, diag.Errorf(n.Max.Span(),
			"iterate_until max must be positive, got %d", maxVal)
	}
	if maxVal > iterateUntilUnrollLimit {
		return nil, diag.Errorf(n.Max.Span(),
			"iterate_until max %d exceeds v0.11 unroll ceiling of %d; "+
				"large-max iterate_until via OpIterateUntilLoop is a planned follow-up",
			maxVal, iterateUntilUnrollLimit)
	}
	return initTy, nil
}

// checkEach handles each(v, f) -> unit. As of v0.13 the legacy
// allowlist of intrinsic names is gone: any function whose effects
// are a subset of the enclosing function's effects is accepted, so
// long as it takes a single argument matching the tensor element
// type. In particular, an `@io` callee requires an `@io` caller.
func (c *checkCtx) checkEach(n *ast.Each) (*Type, error) {
	vTy, err := c.checkExpr(n.V)
	if err != nil {
		return nil, err
	}
	if !vTy.IsTensor() || vTy.Rank() != 1 {
		return nil, diag.Errorf(n.V.Span(),
			"each requires a rank-1 tensor, got %s", vTy)
	}
	if vTy.Shape[0].Var != "" {
		return nil, diag.Errorf(n.V.Span(),
			"each requires a tensor with a concrete length")
	}
	count := vTy.Shape[0].Val
	if count <= 0 {
		return nil, diag.Errorf(n.V.Span(),
			"each tensor must have positive length, got %d", count)
	}
	if count > rangeMaxLen {
		return nil, diag.Errorf(n.V.Span(),
			"each length %d exceeds the v0.11 ceiling of %d",
			count, rangeMaxLen)
	}
	ident, ok := n.Fn.(*ast.Ident)
	if !ok {
		return nil, diag.Errorf(n.Fn.Span(),
			"each requires a named function as its second argument")
	}
	sig, ok := c.fnSigs[ident.Name]
	if !ok {
		return nil, diag.Errorf(ident.Span(), "unknown function %q", ident.Name)
	}
	if !sig.Effects.Subset(c.fnEffects) {
		missing := sig.Effects &^ c.fnEffects
		return nil, diag.Errorf(n.Fn.Span(),
			"each: function %q has effect %s but the enclosing function "+
				"%q is not annotated with that effect; add the annotation "+
				"to the caller, e.g. `@io fn %s(...)`.",
			ident.Name, missing, c.fnName, c.fnName)
	}
	if len(sig.Params) != 1 {
		return nil, diag.Errorf(ident.Span(),
			"each function %q must have 1 parameter, has %d", ident.Name, len(sig.Params))
	}
	elemTy := vTy.ElemType()
	if !sig.Params[0].Equal(elemTy) {
		return nil, diag.Errorf(n.Fn.Span(),
			"each: function %q expects %s, but tensor has element type %s",
			ident.Name, sig.Params[0], elemTy)
	}
	return Unit(), nil
}
