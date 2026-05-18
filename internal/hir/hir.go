// Package hir is a typed AST surface used as a stepping stone to CEIR.
//
// v0.1 keeps it deliberately thin: HIR mirrors the AST but carries
// resolved types. Later phases will lower control flow and `let`s here.
package hir

import "github.com/esque-lang/esquec/internal/types"

type File struct {
	Path string
	Fns  []*Fn
}

type Fn struct {
	Name    string
	Params  []Param
	RetType *types.Type
	Body    Expr
}

type Param struct {
	Name string
	Type *types.Type
}

type Expr interface {
	Type() *types.Type
}

type IntLit struct {
	V int64
	T *types.Type
}

func (e *IntLit) Type() *types.Type { return e.T }

type BoolLit struct {
	V bool
	T *types.Type
}

func (e *BoolLit) Type() *types.Type { return e.T }

type Ident struct {
	Name string
	T    *types.Type
}

func (e *Ident) Type() *types.Type { return e.T }

type BinOp struct {
	Op   string
	L, R Expr
	T    *types.Type
}

func (e *BinOp) Type() *types.Type { return e.T }

type Unary struct {
	Op string
	X  Expr
	T  *types.Type
}

func (e *Unary) Type() *types.Type { return e.T }

type Call struct {
	FnName string
	Args   []Expr
	T      *types.Type
}

func (e *Call) Type() *types.Type { return e.T }

type Block struct {
	Stmts  []Stmt
	Result Expr // may be nil for unit
	T      *types.Type
}

func (e *Block) Type() *types.Type { return e.T }

type If struct {
	Cond Expr
	Then Expr
	Else Expr // may be nil
	T    *types.Type
}

func (e *If) Type() *types.Type { return e.T }

type Stmt interface {
	stmtNode()
}

type LetStmt struct {
	Name string
	Init Expr
	T    *types.Type
}

func (*LetStmt) stmtNode() {}

type ExprStmt struct {
	X Expr
}

func (*ExprStmt) stmtNode() {}

type ReturnStmt struct {
	Value Expr // may be nil
}

func (*ReturnStmt) stmtNode() {}
