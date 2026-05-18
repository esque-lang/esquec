---
title: Parser
---

# Parser

`internal/parse`. Recursive descent with a Pratt-style operator
precedence climber for expressions.

## Entry points

- `parse.File(filename, src) (*ast.File, error)` — full file.
- `parse.Expr(...)` — single expression (used by tests).

A `*ast.File` is a list of `*ast.FnDecl`s.

## Strategy

Top-level parsing is straightforward recursive descent: see
`parseFile` → `parseFnDecl` → `parseParams` etc. Expressions go
through `parseExpr` → `parseBinary(prec)` which is a textbook Pratt
loop on operator precedences (see
[Operators](../language/operators)).

Postfix forms (calls, casts, index access in the future) go through
`parsePostfix`. Builtin calls (`tabulate`, `scan`, `iterate_until`,
`each`, `iterate`) are dispatched by name in `parsePostfix` and
constructed as their dedicated AST nodes rather than as generic
`Call` nodes.

## AST node types

`internal/ast/ast.go`. The major ones:

- **Declarations**: `FnDecl`.
- **Statements**: `LetStmt`.
- **Expressions**: `IntLit` (with `TypeSuffix`), `FloatLit`,
  `BoolLit`, `Ident`, `BinOp`, `Unary`, `Call`, `Lambda`, `Block`,
  `If`, `Match`, `Pipeline`, `Reduce`, `TensorLit`, `RangeExpr`,
  `Tabulate`, `Scan`, `IterateUntil`, `Iterate`, `Each`, `As`.
- **Types**: `NamedType`, `TensorType`.
- **Patterns**: `IntPat`, `BoolPat`, `BindingPat`, `WildcardPat`.

Every AST node carries a `Span` and implements `exprNode()` /
`stmtNode()` / `declNode()` for type discrimination.

## Diagnostics

Parse errors return `*diag.Diagnostic` with the offending span and
a one-line title. Where applicable, they carry a "help" line
pointing at the expected form.
