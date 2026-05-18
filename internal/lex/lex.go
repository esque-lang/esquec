// Package lex implements the esque lexer. v0.0 covers a small subset:
// idents, integer literals, the keyword `fn`, and the punctuation needed
// for `fn main() -> i32 = 42`.
package lex

import (
	"fmt"
	"unicode"
	"unicode/utf8"

	"github.com/esque-lang/esquec/internal/diag"
)

type Lexer struct {
	file string
	src  []byte
	off  int
	line int
	col  int
}

func New(file string, src []byte) *Lexer {
	return &Lexer{file: file, src: src, line: 1, col: 1}
}

func (l *Lexer) pos() diag.Pos {
	return diag.Pos{File: l.file, Line: l.line, Col: l.col, ByteOff: l.off}
}

func (l *Lexer) peek() byte {
	if l.off >= len(l.src) {
		return 0
	}
	return l.src[l.off]
}

func (l *Lexer) advance() byte {
	if l.off >= len(l.src) {
		return 0
	}
	c := l.src[l.off]
	l.off++
	if c == '\n' {
		l.line++
		l.col = 1
	} else {
		l.col++
	}
	return c
}

func (l *Lexer) skipWhitespaceAndComments() error {
	for l.off < len(l.src) {
		c := l.peek()
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			l.advance()
		case c == '#':
			// `#` opens a line comment that runs to the end of the line.
			// Line comments switched from `//` to `#` in v0.14 so that
			// `//` is free as the divide-reduction operator.
			for l.off < len(l.src) && l.peek() != '\n' {
				l.advance()
			}
		case c == '/' && l.off+1 < len(l.src) && l.src[l.off+1] == '*':
			start := l.pos()
			l.advance()
			l.advance()
			depth := 1
			for depth > 0 && l.off < len(l.src) {
				if l.peek() == '/' && l.off+1 < len(l.src) && l.src[l.off+1] == '*' {
					l.advance()
					l.advance()
					depth++
				} else if l.peek() == '*' && l.off+1 < len(l.src) && l.src[l.off+1] == '/' {
					l.advance()
					l.advance()
					depth--
				} else {
					l.advance()
				}
			}
			if depth != 0 {
				return diag.Errorf(diag.Span{Start: start, End: l.pos()}, "unterminated block comment")
			}
		default:
			return nil
		}
	}
	return nil
}

// Next returns the next token.
func (l *Lexer) Next() (Token, error) {
	if err := l.skipWhitespaceAndComments(); err != nil {
		return Token{}, err
	}
	start := l.pos()
	if l.off >= len(l.src) {
		return Token{Kind: TkEOF, Span: diag.Span{Start: start, End: start}}, nil
	}

	c := l.peek()

	// string literal
	if c == '"' {
		return l.lexString(start)
	}

	// char literal
	if c == '\'' {
		return l.lexChar(start)
	}

	// identifier or keyword
	if isIdentStart(c) {
		begin := l.off
		for l.off < len(l.src) && isIdentCont(l.peek()) {
			l.advance()
		}
		lit := string(l.src[begin:l.off])
		span := diag.Span{Start: start, End: l.pos()}
		if k, ok := keywords[lit]; ok {
			return Token{Kind: k, Lit: lit, Span: span}, nil
		}
		return Token{Kind: TkIdent, Lit: lit, Span: span}, nil
	}

	// numeric literal (int or float)
	if c >= '0' && c <= '9' {
		begin := l.off
		isFloat := false
		for l.off < len(l.src) {
			c := l.peek()
			if c >= '0' && c <= '9' {
				l.advance()
			} else if c == '_' && l.off+1 < len(l.src) &&
				l.src[l.off+1] >= '0' && l.src[l.off+1] <= '9' {
				// digit separator only between digits; a trailing `_`
				// belongs to the type suffix (`_i32`, `_f64`, ...).
				l.advance()
			} else if c == '.' && l.off+1 < len(l.src) {
				next := l.src[l.off+1]
				// Check it's not a range operator (..) or method call
				if next >= '0' && next <= '9' {
					isFloat = true
					l.advance() // consume '.'
				} else {
					break
				}
			} else if c == 'e' || c == 'E' {
				// Scientific notation
				isFloat = true
				l.advance()
				if l.peek() == '+' || l.peek() == '-' {
					l.advance()
				}
			} else {
				break
			}
		}
		// Check for type suffix like _f32, _f64, _i32, etc.
		if l.peek() == '_' && l.off+1 < len(l.src) {
			next := l.src[l.off+1]
			if next == 'f' || next == 'i' || next == 'u' {
				l.advance() // consume '_'
				for l.off < len(l.src) && isIdentCont(l.peek()) {
					l.advance()
				}
				// Check if it's a float suffix
				lit := string(l.src[begin:l.off])
				if len(lit) > 3 && (lit[len(lit)-3:] == "f32" || lit[len(lit)-3:] == "f64") {
					isFloat = true
				}
			}
		}
		lit := string(l.src[begin:l.off])
		if isFloat {
			return Token{Kind: TkFloat, Lit: lit, Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkInt, Lit: lit, Span: diag.Span{Start: start, End: l.pos()}}, nil
	}

	// punctuation / operators
	switch c {
	case '(':
		l.advance()
		return Token{Kind: TkLParen, Lit: "(", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case ')':
		l.advance()
		return Token{Kind: TkRParen, Lit: ")", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '{':
		l.advance()
		return Token{Kind: TkLBrace, Lit: "{", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '}':
		l.advance()
		return Token{Kind: TkRBrace, Lit: "}", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '[':
		l.advance()
		return Token{Kind: TkLBracket, Lit: "[", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case ']':
		l.advance()
		return Token{Kind: TkRBracket, Lit: "]", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '=':
		l.advance()
		if l.peek() == '=' {
			l.advance()
			return Token{Kind: TkEqEq, Lit: "==", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		if l.peek() == '>' {
			l.advance()
			return Token{Kind: TkFatArrow, Lit: "=>", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkEq, Lit: "=", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '!':
		l.advance()
		if l.peek() == '=' {
			l.advance()
			return Token{Kind: TkBangEq, Lit: "!=", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkBang, Lit: "!", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '<':
		l.advance()
		if l.peek() == '=' {
			l.advance()
			return Token{Kind: TkLtEq, Lit: "<=", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkLt, Lit: "<", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '>':
		l.advance()
		if l.peek() == '=' {
			l.advance()
			return Token{Kind: TkGtEq, Lit: ">=", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkGt, Lit: ">", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case ';':
		l.advance()
		return Token{Kind: TkSemi, Lit: ";", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case ':':
		l.advance()
		return Token{Kind: TkColon, Lit: ":", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case ',':
		l.advance()
		return Token{Kind: TkComma, Lit: ",", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '+':
		l.advance()
		if l.peek() == '/' {
			l.advance()
			return Token{Kind: TkPlusSlash, Lit: "+/", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkPlus, Lit: "+", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '@':
		l.advance()
		// `@<ident>` attached (no whitespace between `@` and the ident
		// start) lexes as a single TkAttr token whose Lit is the
		// attribute name. This disambiguates attributes from the
		// matmul operator: `a @ b` (with surrounding whitespace) is
		// matmul, while `@io fn` / `@kernel fn` are attribute leads.
		if l.off < len(l.src) && isIdentStart(l.src[l.off]) {
			nameStart := l.off
			for l.off < len(l.src) && isIdentCont(l.peek()) {
				l.advance()
			}
			name := string(l.src[nameStart:l.off])
			if name == "kernel" {
				return Token{Kind: TkKernel, Lit: "@kernel", Span: diag.Span{Start: start, End: l.pos()}}, nil
			}
			return Token{Kind: TkAttr, Lit: name, Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkAt, Lit: "@", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '*':
		l.advance()
		if l.peek() == '/' {
			l.advance()
			return Token{Kind: TkStarSlash, Lit: "*/", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkStar, Lit: "*", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '.':
		l.advance()
		switch l.peek() {
		case '+':
			l.advance()
			return Token{Kind: TkDotPlus, Lit: ".+", Span: diag.Span{Start: start, End: l.pos()}}, nil
		case '-':
			l.advance()
			return Token{Kind: TkDotMinus, Lit: ".-", Span: diag.Span{Start: start, End: l.pos()}}, nil
		case '*':
			l.advance()
			return Token{Kind: TkDotStar, Lit: ".*", Span: diag.Span{Start: start, End: l.pos()}}, nil
		case '/':
			l.advance()
			return Token{Kind: TkDotSlash, Lit: "./", Span: diag.Span{Start: start, End: l.pos()}}, nil
		case '%':
			l.advance()
			return Token{Kind: TkDotPercent, Lit: ".%", Span: diag.Span{Start: start, End: l.pos()}}, nil
		case '.':
			l.advance()
			if l.peek() == '=' {
				l.advance()
				return Token{Kind: TkDotDotEq, Lit: "..=", Span: diag.Span{Start: start, End: l.pos()}}, nil
			}
			return Token{Kind: TkDotDot, Lit: "..", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		// Single dot - could be method access, but not supported yet
		return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()}, "unexpected character '.'")

	case '/':
		l.advance()
		if l.peek() == '/' {
			l.advance()
			return Token{Kind: TkSlashSlash, Lit: "//", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkSlash, Lit: "/", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '%':
		l.advance()
		return Token{Kind: TkPercent, Lit: "%", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '&':
		l.advance()
		if l.peek() == '&' {
			l.advance()
			return Token{Kind: TkAmpAmp, Lit: "&&", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		// Single & not yet supported, fall through to error
	case '|':
		l.advance()
		if l.peek() == '|' {
			l.advance()
			return Token{Kind: TkPipePipe, Lit: "||", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		if l.peek() == '>' {
			l.advance()
			return Token{Kind: TkPipe, Lit: "|>", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		// Single | for lambda syntax: |x, y| expr
		return Token{Kind: TkBar, Lit: "|", Span: diag.Span{Start: start, End: l.pos()}}, nil
	case '-':
		l.advance()
		if l.peek() == '>' {
			l.advance()
			return Token{Kind: TkArrow, Lit: "->", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		if l.peek() == '/' {
			l.advance()
			return Token{Kind: TkMinusSlash, Lit: "-/", Span: diag.Span{Start: start, End: l.pos()}}, nil
		}
		return Token{Kind: TkMinus, Lit: "-", Span: diag.Span{Start: start, End: l.pos()}}, nil
	}

	// unknown rune. Advance past it so subsequent calls can return more
	// tokens (enables parser error recovery to make progress).
	r, sz := utf8.DecodeRune(l.src[l.off:])
	if sz < 1 {
		sz = 1
	}
	for i := 0; i < sz; i++ {
		l.advance()
	}
	end := l.pos()
	return Token{}, diag.Errorf(diag.Span{Start: start, End: end},
		"unexpected character %q", fmt.Sprintf("%c", r))
}

// lexString consumes a "..." string literal with simple escapes.
// Recognised escapes: \\ \" \n \r \t \0.
func (l *Lexer) lexString(start diag.Pos) (Token, error) {
	l.advance() // opening "
	var buf []byte
	for {
		if l.off >= len(l.src) {
			return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
				"unterminated string literal")
		}
		c := l.peek()
		if c == '"' {
			l.advance()
			return Token{
				Kind: TkString,
				Lit:  string(buf),
				Span: diag.Span{Start: start, End: l.pos()},
			}, nil
		}
		if c == '\n' {
			return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
				"unterminated string literal (newline)")
		}
		if c == '\\' {
			l.advance()
			if l.off >= len(l.src) {
				return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
					"unterminated escape in string")
			}
			esc := l.advance()
			switch esc {
			case '\\':
				buf = append(buf, '\\')
			case '"':
				buf = append(buf, '"')
			case 'n':
				buf = append(buf, '\n')
			case 'r':
				buf = append(buf, '\r')
			case 't':
				buf = append(buf, '\t')
			case '0':
				buf = append(buf, 0)
			default:
				return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
					"unknown escape \\%c in string", esc)
			}
			continue
		}
		buf = append(buf, c)
		l.advance()
	}
}

// lexChar consumes a '.' character literal. Stores the codepoint in Lit as
// a decimal string so the parser can promote it to an int constant.
func (l *Lexer) lexChar(start diag.Pos) (Token, error) {
	l.advance() // opening '
	if l.off >= len(l.src) {
		return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
			"unterminated char literal")
	}
	var r rune
	c := l.peek()
	if c == '\\' {
		l.advance()
		if l.off >= len(l.src) {
			return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
				"unterminated escape in char")
		}
		esc := l.advance()
		switch esc {
		case '\\':
			r = '\\'
		case '\'':
			r = '\''
		case 'n':
			r = '\n'
		case 'r':
			r = '\r'
		case 't':
			r = '\t'
		case '0':
			r = 0
		default:
			return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
				"unknown escape \\%c in char", esc)
		}
	} else {
		// Decode a UTF-8 rune from the source.
		rr, size := utf8.DecodeRune(l.src[l.off:])
		r = rr
		for i := 0; i < size; i++ {
			l.advance()
		}
	}
	if l.off >= len(l.src) || l.peek() != '\'' {
		return Token{}, diag.Errorf(diag.Span{Start: start, End: l.pos()},
			"expected closing ' in char literal")
	}
	l.advance() // closing '
	return Token{
		Kind: TkChar,
		Lit:  fmt.Sprintf("%d", r),
		Span: diag.Span{Start: start, End: l.pos()},
	}, nil
}

func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		(c >= 0x80 && unicode.IsLetter(rune(c)))
}

func isIdentCont(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}
