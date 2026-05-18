// Package ast defines the abstract syntax tree.
package ast

import "github.com/esque-lang/esquec/internal/diag"

// File is a parsed source file.
type File struct {
	Path  string
	Items []Item
}

// Item is a top-level declaration.
type Item interface {
	itemNode()
	Span() diag.Span
}

// FnDecl is a function declaration.
//
// Two forms:
//   fn NAME(PARAMS) -> RET = EXPR
//   fn NAME(PARAMS) -> RET { STMTS; EXPR }
//
// With shape parameters:
//   fn NAME[SHAPE_PARAMS](PARAMS) -> RET = EXPR
type FnDecl struct {
	NameTok     diag.Span
	Name        string
	ShapeParams []ShapeParam // shape parameters [N, M: nat]
	Params      []Param
	RetType     TypeExpr
	Body        Expr // either a single expr or a Block
	SpanRng     diag.Span
	Attrs       []Attr // attributes attached to the fn (e.g. @kernel, @grad)
	IsKernel    bool   // true if Attrs contains an attribute named "kernel"
}

// Attr is an attribute attached to an item: @name or @name(arg, ...).
//
// The attribute parser is uniform — it does not know which names are
// meaningful. Type-checking and lowering passes consult Attrs to drive
// behaviour (e.g. `kernel` flags a GPU entry point, `grad` schedules
// autodiff, `inline` advises the inliner).
type Attr struct {
	Name    string
	Args    []Expr // empty when no parentheses are present
	SpanRng diag.Span
}

func (a Attr) Span() diag.Span { return a.SpanRng }

// ShapeParam is a shape parameter declaration: N or N: nat
type ShapeParam struct {
	Name string
	Span diag.Span
}

func (*FnDecl) itemNode()        {}
func (f *FnDecl) Span() diag.Span { return f.SpanRng }

// Param is a function parameter.
type Param struct {
	Name string
	Type TypeExpr
	Span diag.Span
}

// TypeExpr is a type expression.
type TypeExpr interface {
	typeNode()
	Span() diag.Span
}

// NamedType references a named type like `i32`.
type NamedType struct {
	Name    string
	SpanRng diag.Span
}

func (*NamedType) typeNode()          {}
func (t *NamedType) Span() diag.Span { return t.SpanRng }

// TensorType is a shaped tensor type: f32[N, M]
type TensorType struct {
	Elem    TypeExpr    // element type (e.g., f32)
	Shape   []ShapeExpr // dimensions
	SpanRng diag.Span
}

func (*TensorType) typeNode()          {}
func (t *TensorType) Span() diag.Span { return t.SpanRng }

// ShapeExpr is a shape dimension expression.
type ShapeExpr interface {
	shapeNode()
	Span() diag.Span
}

// ShapeLit is an integer shape constant: f32[3, 4]
type ShapeLit struct {
	Value   int64
	SpanRng diag.Span
}

func (*ShapeLit) shapeNode()          {}
func (s *ShapeLit) Span() diag.Span { return s.SpanRng }

// ShapeVar is a shape variable: f32[N, M]
type ShapeVar struct {
	Name    string
	SpanRng diag.Span
}

func (*ShapeVar) shapeNode()          {}
func (s *ShapeVar) Span() diag.Span { return s.SpanRng }

// ShapeBinOp is a shape arithmetic expression: f32[N+1, 2*M]
type ShapeBinOp struct {
	Op      string // "+", "-", "*", "/"
	L, R    ShapeExpr
	SpanRng diag.Span
}

func (*ShapeBinOp) shapeNode()          {}
func (s *ShapeBinOp) Span() diag.Span { return s.SpanRng }

// Expr is an expression node.
type Expr interface {
	exprNode()
	Span() diag.Span
}

// IntLit is an integer literal.
//
// TypeSuffix records an explicit type suffix from the source (e.g. "i32",
// "i64", "u8"). Empty means the literal had no suffix and defaults to i32.
type IntLit struct {
	Value      int64
	TypeSuffix string
	SpanRng    diag.Span
}

func (*IntLit) exprNode()          {}
func (e *IntLit) Span() diag.Span { return e.SpanRng }

// FloatLit is a floating-point literal.
type FloatLit struct {
	Value   float64
	Width   int // 32 or 64, default 32
	SpanRng diag.Span
}

func (*FloatLit) exprNode()          {}
func (e *FloatLit) Span() diag.Span { return e.SpanRng }

// BoolLit is a boolean literal (true/false).
type BoolLit struct {
	Value   bool
	SpanRng diag.Span
}

func (*BoolLit) exprNode()          {}
func (e *BoolLit) Span() diag.Span { return e.SpanRng }

// StringLit is a string literal. Value holds the unescaped UTF-8
// bytes (the lexer has already processed escape sequences). v0.12
// strings are immutable fat pointers; the only operation defined on
// them is being passed to print_str.
type StringLit struct {
	Value   string
	SpanRng diag.Span
}

func (*StringLit) exprNode()         {}
func (e *StringLit) Span() diag.Span { return e.SpanRng }

// Ident is an identifier expression.
type Ident struct {
	Name    string
	SpanRng diag.Span
}

func (*Ident) exprNode()          {}
func (e *Ident) Span() diag.Span { return e.SpanRng }

// BinOp is a binary operation.
type BinOp struct {
	Op      string
	L, R    Expr
	SpanRng diag.Span
}

func (*BinOp) exprNode()          {}
func (e *BinOp) Span() diag.Span { return e.SpanRng }

// Unary is a prefix unary operation (e.g. negation).
type Unary struct {
	Op      string
	X       Expr
	SpanRng diag.Span
}

func (*Unary) exprNode()          {}
func (e *Unary) Span() diag.Span { return e.SpanRng }

// Call is a function call expression: f(a, b, c) or f[N, M](a, b, c)
type Call struct {
	Fn        Expr
	ShapeArgs []ShapeExpr // optional shape arguments [N, M]
	Args      []Expr
	SpanRng   diag.Span
}

func (*Call) exprNode()          {}
func (e *Call) Span() diag.Span { return e.SpanRng }

// Pipeline is a pipeline expression: x |> f |> g
type Pipeline struct {
	L, R    Expr
	SpanRng diag.Span
}

func (*Pipeline) exprNode()          {}
func (e *Pipeline) Span() diag.Span { return e.SpanRng }

// Reduce is a reduction expression: +/ arr, -/ arr, */ arr, // arr.
type Reduce struct {
	Op      string // "+", "-", "*", "/", "max", "min"
	X       Expr
	SpanRng diag.Span
}

func (*Reduce) exprNode()          {}
func (e *Reduce) Span() diag.Span { return e.SpanRng }

// Fold is a fold/reduce expression with custom function: f/tensor or f/init/tensor
type Fold struct {
	Fn      Expr   // The reducing function (identifier or lambda)
	Init    Expr   // Initial value (nil for reduce without init)
	X       Expr   // The tensor to fold over
	SpanRng diag.Span
}

func (*Fold) exprNode()          {}
func (e *Fold) Span() diag.Span { return e.SpanRng }

// Iterate is a stateful iteration: iterate(n, init, f)
type Iterate struct {
	N       Expr   // Number of iterations
	Init    Expr   // Initial value
	Fn      Expr   // Update function (takes state, returns new state)
	SpanRng diag.Span
}

func (*Iterate) exprNode()          {}
func (e *Iterate) Span() diag.Span { return e.SpanRng }

// RangeExpr is an integer range: lo..hi (exclusive) or lo..=hi (inclusive).
//
// In v0.10 both bounds must be integer literals; the range desugars to
// a tensor literal of i32 values during CEIR lowering.
type RangeExpr struct {
	Lo, Hi    Expr
	Inclusive bool
	SpanRng   diag.Span
}

func (*RangeExpr) exprNode()         {}
func (e *RangeExpr) Span() diag.Span { return e.SpanRng }

// Tabulate is index-driven tensor construction: tabulate(n, |i| f(i)).
type Tabulate struct {
	N       Expr // must reduce to a literal int by type-check
	Fn      Expr // 1-arg lambda or named fn
	SpanRng diag.Span
}

func (*Tabulate) exprNode()         {}
func (e *Tabulate) Span() diag.Span { return e.SpanRng }

// Scan is a prefix accumulator: scan(init, |a, x| f(a, x), v) -> typeof(v).
type Scan struct {
	Init, Fn, V Expr
	SpanRng     diag.Span
}

func (*Scan) exprNode()         {}
func (e *Scan) Span() diag.Span { return e.SpanRng }

// IterateUntil is a bounded fixpoint:
// iterate_until(init, |s| step(s), |s| pred(s), max).
//
// Returns the first state s for which pred(s) is true, or the state
// after `max` iterations if pred never becomes true.
type IterateUntil struct {
	Init, Step, Pred Expr
	Max              Expr // must reduce to a literal int by type-check
	SpanRng          diag.Span
}

func (*IterateUntil) exprNode()         {}
func (e *IterateUntil) Span() diag.Span { return e.SpanRng }

// Each is side-effect iteration over a tensor: each(v, f).
//
// As of v0.13, f must be a named function whose effect set is a subset
// of the enclosing function's effects: an `@io` callee requires an
// `@io` caller. A pure callee is accepted in any context.
type Each struct {
	V, Fn   Expr
	SpanRng diag.Span
}

func (*Each) exprNode()         {}
func (e *Each) Span() diag.Span { return e.SpanRng }

// Lambda is an anonymous function: |x, y| x + y
type Lambda struct {
	Params  []LambdaParam
	Body    Expr
	SpanRng diag.Span
}

func (*Lambda) exprNode()          {}
func (e *Lambda) Span() diag.Span { return e.SpanRng }

// LambdaParam is a lambda parameter (may be untyped)
type LambdaParam struct {
	Name string
	Type TypeExpr // nil if inferred
	Span diag.Span
}

// TensorLit is a tensor literal: [1.0, 2.0, 3.0]
type TensorLit struct {
	Elems   []Expr
	SpanRng diag.Span
}

func (*TensorLit) exprNode()          {}
func (e *TensorLit) Span() diag.Span { return e.SpanRng }

// Cast is a type cast expression: x as i32
type Cast struct {
	X       Expr
	Target  TypeExpr
	SpanRng diag.Span
}

func (*Cast) exprNode()          {}
func (e *Cast) Span() diag.Span { return e.SpanRng }

// Block is a braced sequence of statements ending with an expression.
// { let x = 1; let y = 2; x + y }
type Block struct {
	Stmts   []Stmt
	Result  Expr // trailing expression; may be nil for unit
	SpanRng diag.Span
}

func (*Block) exprNode()          {}
func (e *Block) Span() diag.Span { return e.SpanRng }

// If is an if-else expression: if cond { a } else { b }
type If struct {
	Cond    Expr
	Then    Expr
	Else    Expr // may be nil
	SpanRng diag.Span
}

func (*If) exprNode()          {}
func (e *If) Span() diag.Span { return e.SpanRng }

// Match is a match expression: match x { Pat => expr, ... }
type Match struct {
	Scrutinee Expr
	Arms      []MatchArm
	SpanRng   diag.Span
}

func (*Match) exprNode()          {}
func (e *Match) Span() diag.Span { return e.SpanRng }

// MatchArm is a single arm of a match expression.
type MatchArm struct {
	Pattern Pattern
	Guard   Expr // optional: `if cond` after pattern
	Body    Expr
	SpanRng diag.Span
}

// Pattern is a match pattern.
type Pattern interface {
	patternNode()
	Span() diag.Span
}

// WildcardPat is the `_` pattern that matches anything.
type WildcardPat struct {
	SpanRng diag.Span
}

func (*WildcardPat) patternNode()        {}
func (p *WildcardPat) Span() diag.Span { return p.SpanRng }

// LitPat is a literal pattern (integer or boolean).
type LitPat struct {
	Value   Expr // IntLit or BoolLit
	SpanRng diag.Span
}

func (*LitPat) patternNode()        {}
func (p *LitPat) Span() diag.Span { return p.SpanRng }

// BindPat is a binding pattern that binds the matched value to a name.
type BindPat struct {
	Name    string
	SpanRng diag.Span
}

func (*BindPat) patternNode()        {}
func (p *BindPat) Span() diag.Span { return p.SpanRng }

// Stmt is a statement within a block.
type Stmt interface {
	stmtNode()
	Span() diag.Span
}

// LetStmt is a let binding: let x: T = expr
type LetStmt struct {
	Name    string
	Type    TypeExpr // may be nil (type inferred)
	Init    Expr
	SpanRng diag.Span
}

func (*LetStmt) stmtNode()          {}
func (s *LetStmt) Span() diag.Span { return s.SpanRng }

// ExprStmt is an expression used as a statement (discards value).
type ExprStmt struct {
	X       Expr
	SpanRng diag.Span
}

func (*ExprStmt) stmtNode()          {}
func (s *ExprStmt) Span() diag.Span { return s.SpanRng }

// ReturnStmt is `return expr`.
type ReturnStmt struct {
	Value   Expr // may be nil for unit return
	SpanRng diag.Span
}

func (*ReturnStmt) stmtNode()          {}
func (s *ReturnStmt) Span() diag.Span { return s.SpanRng }
