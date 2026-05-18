package lex

import "github.com/esque-lang/esquec/internal/diag"

// Kind identifies a token type.
type Kind int

const (
	TkEOF Kind = iota
	TkIdent
	TkInt
	TkFloat
	TkString
	TkChar

	// keywords
	TkFn
	TkReturn
	TkLet
	TkMut
	TkIf
	TkElse
	TkMatch
	TkTrue
	TkFalse
	TkAs
	TkKernel // @kernel annotation (legacy single-token form)
	TkIn     // in

	// punctuation / operators
	TkLParen
	TkRParen
	TkLBrace
	TkRBrace
	TkLBracket // [
	TkRBracket // ]
	TkArrow    // ->
	TkFatArrow // =>
	TkEq       // =
	TkEqEq     // ==
	TkBangEq   // !=
	TkLt       // <
	TkLtEq     // <=
	TkGt       // >
	TkGtEq     // >=
	TkSemi     // ;
	TkColon    // :
	TkComma    // ,
	TkPlus     // +
	TkMinus    // -
	TkStar     // *
	TkSlash    // /
	TkPercent  // %
	TkAmpAmp   // &&
	TkPipePipe // ||
	TkBang     // !

	// Tensor-first operators
	TkPipe       // |>
	TkBar        // | (single bar for lambdas)
	TkAt         // @ (matmul; bare with surrounding whitespace)
	TkAttr       // @<ident> attribute lead, attached form (e.g. @io, @grad). Lit holds the name.
	TkDotPlus    // .+
	TkDotMinus   // .-
	TkDotStar    // .*
	TkDotSlash   // ./
	TkDotPercent // .%
	TkPlusSlash  // +/
	TkMinusSlash // -/
	TkStarSlash  // */
	TkSlashSlash // //  (reduce-by-divide; line comments are `#`)
	TkDotDot     // ..
	TkDotDotEq   // ..=
)

// Token is a lexed token with source span.
type Token struct {
	Kind Kind
	Lit  string
	Span diag.Span
}

func (k Kind) String() string {
	switch k {
	case TkEOF:
		return "eof"
	case TkIdent:
		return "ident"
	case TkInt:
		return "int"
	case TkFloat:
		return "float"
	case TkString:
		return "string"
	case TkChar:
		return "char"
	case TkFn:
		return "fn"
	case TkReturn:
		return "return"
	case TkLet:
		return "let"
	case TkMut:
		return "mut"
	case TkIf:
		return "if"
	case TkElse:
		return "else"
	case TkMatch:
		return "match"
	case TkTrue:
		return "true"
	case TkFalse:
		return "false"
	case TkAs:
		return "as"
	case TkKernel:
		return "@kernel"
	case TkIn:
		return "in"
	case TkLParen:
		return "("
	case TkRParen:
		return ")"
	case TkLBrace:
		return "{"
	case TkRBrace:
		return "}"
	case TkLBracket:
		return "["
	case TkRBracket:
		return "]"
	case TkArrow:
		return "->"
	case TkFatArrow:
		return "=>"
	case TkEq:
		return "="
	case TkEqEq:
		return "=="
	case TkBangEq:
		return "!="
	case TkLt:
		return "<"
	case TkLtEq:
		return "<="
	case TkGt:
		return ">"
	case TkGtEq:
		return ">="
	case TkSemi:
		return ";"
	case TkColon:
		return ":"
	case TkComma:
		return ","
	case TkPlus:
		return "+"
	case TkMinus:
		return "-"
	case TkStar:
		return "*"
	case TkSlash:
		return "/"
	case TkPercent:
		return "%"
	case TkAmpAmp:
		return "&&"
	case TkPipePipe:
		return "||"
	case TkBang:
		return "!"
	case TkPipe:
		return "|>"
	case TkBar:
		return "|"
	case TkAt:
		return "@"
	case TkAttr:
		return "@<attr>"
	case TkDotPlus:
		return ".+"
	case TkDotMinus:
		return ".-"
	case TkDotStar:
		return ".*"
	case TkDotSlash:
		return "./"
	case TkDotPercent:
		return ".%"
	case TkPlusSlash:
		return "+/"
	case TkMinusSlash:
		return "-/"
	case TkStarSlash:
		return "*/"
	case TkSlashSlash:
		return "//"
	case TkDotDot:
		return ".."
	case TkDotDotEq:
		return "..="
	}
	return "unknown"
}

var keywords = map[string]Kind{
	"fn":     TkFn,
	"return": TkReturn,
	"let":    TkLet,
	"mut":    TkMut,
	"if":     TkIf,
	"else":   TkElse,
	"match":  TkMatch,
	"true":   TkTrue,
	"false":  TkFalse,
	"as":     TkAs,
	"in":     TkIn,
}
