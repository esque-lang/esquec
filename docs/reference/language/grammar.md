---
title: Grammar (EBNF)
---

# Grammar (EBNF)

This is the implemented v0.13 grammar in (mostly) EBNF. Items that
are parsed but not yet codegened (e.g. `@` matmul) are included.
Items in the spec but not yet parsed are *not* included here — see
[Specification](/reference/spec/overview) for those.

```ebnf
File           = { FnDecl } ;

FnDecl         = "fn" Ident [ ShapeParams ] "(" Params ")" "->" Type
                 ( "=" Expr | Block ) ;
ShapeParams    = "[" ShapeParam { "," ShapeParam } "]" ;
ShapeParam     = Ident [ ":" "nat" ] ;
Params         = [ Param { "," Param } ] ;
Param          = Ident ":" Type ;

Type           = ElemType [ "[" ShapeList "]" ] ;
ElemType       = "i8" | "i16" | "i32" | "i64"
               | "u8" | "u16" | "u32" | "u64"
               | "f32" | "f64"
               | "bool" | "unit" ;
ShapeList      = ShapeExpr { "," ShapeExpr } ;
ShapeExpr      = ShapeTerm { ( "+" | "-" ) ShapeTerm } ;
ShapeTerm      = ShapeFactor { ( "*" | "/" ) ShapeFactor } ;
ShapeFactor    = Int | Ident | "(" ShapeExpr ")" ;

Block          = "{" { Stmt ";" } [ Expr ] "}" ;
Stmt           = LetStmt | Expr ;
LetStmt        = "let" Ident [ ":" Type ] "=" Expr ;

Expr           = Pratt ;       (* see Operators page for table *)

Lambda         = "|" [ Ident { "," Ident } ] "|" Expr ;

Primary        = IntLit
               | FloatLit
               | BoolLit
               | CharLit
               | TensorLit
               | RangeExpr
               | Lambda
               | Ident [ "(" Args ")" ]
               | If
               | Match
               | Block
               | "(" Expr ")"
               ;
TensorLit      = "[" [ Expr { "," Expr } ] "]" ;
RangeExpr      = Expr ( ".." | "..=" ) Expr ;          (* prec 15 *)

If             = "if" Expr Block [ "else" ( Block | If ) ] ;
Match          = "match" Expr "{" MatchArm { "," MatchArm } [ "," ] "}" ;
MatchArm       = Pattern [ "if" Expr ] "=>" Expr ;
Pattern        = "_" | Ident | IntLit | CharLit | "true" | "false"
               | "-" IntLit ;

Args           = [ Expr { "," Expr } ] ;

(* Built-in call dispatch (all are postfix calls):
   tabulate   ( N , Expr )
   scan       ( Expr , Expr , Expr )
   iterate_until ( Expr , Expr , Expr , N )
   each       ( Expr , Ident )
   iterate    ( N , Expr , Expr )
*)

(* Postfix forms *)
PostfixCall    = Expr "(" Args ")" ;
PostfixAs      = Expr "as" Type ;

(* Operators (see Operators page for the precedence table) *)
BinOp          = "+" | "-" | "*" | "/" | "%"
               | ".+" | ".-" | ".*" | "./" | ".%"
               | "@" | "|>"
               | "==" | "!=" | "<" | "<=" | ">" | ">="
               | "&&" | "||"
               | ".." | "..="
               ;

(* Reductions (prefix unary) *)
ReducePrefix   = ( "+/" | "-/" | "*/" | "//" ) Expr ;

(* Lexical (informal) *)
IntLit         = digit { digit | "_" } [ IntSuffix ]
               | "0x" hex { hex | "_" } [ IntSuffix ] ;
IntSuffix      = "_i8" | "_i16" | "_i32" | "_i64"
               | "_u8" | "_u16" | "_u32" | "_u64" ;
FloatLit       = digit { digit | "_" } "." { digit | "_" }
                 [ Exponent ] [ FloatSuffix ]
               | digit { digit | "_" } Exponent [ FloatSuffix ] ;
Exponent       = ( "e" | "E" ) [ "+" | "-" ] digit { digit } ;
FloatSuffix    = "_f32" | "_f64" ;
BoolLit        = "true" | "false" ;
CharLit        = "'" Char "'" ;          (* with standard escapes *)
StringLit      = "\"" { Char } "\"" ;    (* immutable UTF-8 fat pointer; v0.12 *)
Ident          = letter { letter | digit | "_" } ;
```

Reserved words: `fn`, `return`, `let`, `if`, `else`, `match`,
`true`, `false`, `in`, `as`, `mut`. (`for`, `while`, `break`,
`continue` are **not** reserved — they parse as identifiers.)
