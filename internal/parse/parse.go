// Package parse implements a recursive-descent parser for esque.
package parse

import (
	"strconv"
	"strings"

	"github.com/esque-lang/esquec/internal/ast"
	"github.com/esque-lang/esquec/internal/diag"
	"github.com/esque-lang/esquec/internal/lex"
)

// parseFloatLit parses a float literal with optional suffix
func parseFloatLit(lit string) (float64, int, error) {
	width := 32 // default
	clean := strings.ReplaceAll(lit, "_", "")

	// Check for type suffix
	if strings.HasSuffix(clean, "f64") {
		width = 64
		clean = clean[:len(clean)-3]
	} else if strings.HasSuffix(clean, "f32") {
		width = 32
		clean = clean[:len(clean)-3]
	}

	v, err := strconv.ParseFloat(clean, 64)
	if err != nil {
		return 0, 0, err
	}
	return v, width, nil
}

// intSuffixes lists every integer type suffix the lexer/parser recognises.
// The type-checker decides which are accepted on the current backend.
var intSuffixes = []string{
	"i8", "i16", "i32", "i64",
	"u8", "u16", "u32", "u64",
}

// parseIntLit parses an integer literal lexeme, returning the value and
// the (possibly empty) type suffix. The lexer keeps the `_` digit
// separator and the suffix in the literal text; this helper strips both.
func parseIntLit(lit string) (int64, string, error) {
	clean := strings.ReplaceAll(lit, "_", "")
	suffix := ""
	for _, s := range intSuffixes {
		if strings.HasSuffix(clean, s) {
			suffix = s
			clean = clean[:len(clean)-len(s)]
			break
		}
	}
	v, err := strconv.ParseInt(clean, 10, 64)
	if err != nil {
		return 0, "", err
	}
	return v, suffix, nil
}

type parser struct {
	lx   *lex.Lexer
	cur  lex.Token
	peek lex.Token
	err  error
	bag  *diag.Bag
}

// File parses an entire source file. On error, it returns the first
// diagnostic encountered. Use FileWithBag to collect multiple diagnostics
// via top-level error recovery.
func File(path string, src []byte) (*ast.File, error) {
	f, bag := FileWithBag(path, src)
	if bag.HasErrors() {
		return nil, bag.First()
	}
	return f, nil
}

// FileWithBag parses an entire source file, collecting all diagnostics
// into a Bag. On a parse error inside an item, the parser synchronizes to
// the next top-level item boundary (`fn` or `@kernel`) and continues, so
// independent errors in separate functions are all reported.
func FileWithBag(path string, src []byte) (*ast.File, *diag.Bag) {
	bag := &diag.Bag{}
	p := &parser{lx: lex.New(path, src), bag: bag}
	if err := p.bump(); err != nil {
		bag.Add(asDiag(err))
		return &ast.File{Path: path}, bag
	}
	if err := p.bump(); err != nil {
		bag.Add(asDiag(err))
		return &ast.File{Path: path}, bag
	}
	f := &ast.File{Path: path}
	for p.cur.Kind != lex.TkEOF {
		// Snapshot start position to detect lack of progress.
		startOff := p.cur.Span.Start.ByteOff
		it, err := p.parseItem()
		if err != nil {
			bag.Add(asDiag(err))
			// If we made no progress, force at least one token of advance
			// so synchronize doesn't loop forever.
			if p.cur.Span.Start.ByteOff == startOff && p.cur.Kind != lex.TkEOF {
				if berr := p.bump(); berr != nil {
					bag.Add(asDiag(berr))
					return f, bag
				}
			}
			if serr := p.synchronize(); serr != nil {
				bag.Add(asDiag(serr))
				return f, bag
			}
			continue
		}
		f.Items = append(f.Items, it)
	}
	return f, bag
}

// asDiag normalizes an error returned from parser internals into a
// *diag.Diagnostic so it can be stored in a Bag without losing structure.
func asDiag(err error) *diag.Diagnostic {
	if d, ok := err.(*diag.Diagnostic); ok {
		return d
	}
	return &diag.Diagnostic{
		Severity: diag.SevError,
		Primary:  diag.Label{Msg: err.Error()},
	}
}

// synchronize discards tokens until the parser is positioned at a likely
// top-level item boundary: `fn`, `@kernel`, or EOF. Returns a non-nil
// error only if the lexer fails.
func (p *parser) synchronize() error {
	for p.cur.Kind != lex.TkEOF && p.cur.Kind != lex.TkFn && p.cur.Kind != lex.TkKernel {
		if err := p.bump(); err != nil {
			return err
		}
	}
	return nil
}

func (p *parser) bump() error {
	p.cur = p.peek
	for {
		t, err := p.lx.Next()
		if err == nil {
			p.peek = t
			return nil
		}
		// In recovery mode (bag attached), record the lex error and keep
		// pulling tokens; the lexer has advanced past the offending rune,
		// so the next call should make progress.
		if p.bag == nil {
			return err
		}
		p.bag.Add(asDiag(err))
	}
}

func (p *parser) expect(k lex.Kind) (lex.Token, error) {
	if p.cur.Kind != k {
		return lex.Token{}, diag.Errorf(p.cur.Span, "expected %s, got %s", k, p.cur.Kind)
	}
	t := p.cur
	if err := p.bump(); err != nil {
		return lex.Token{}, err
	}
	return t, nil
}

func (p *parser) parseItem() (ast.Item, error) {
	// Read any leading attributes: @kernel, @grad, @inline(always), etc.
	// The legacy TkKernel token is treated as `@kernel`.
	attrs, err := p.parseAttrs()
	if err != nil {
		return nil, err
	}

	switch p.cur.Kind {
	case lex.TkFn:
		fn, err := p.parseFn()
		if err != nil {
			return nil, err
		}
		fn.Attrs = attrs
		for _, a := range attrs {
			if a.Name == "kernel" {
				fn.IsKernel = true
				break
			}
		}
		return fn, nil
	}
	if len(attrs) > 0 {
		return nil, diag.Errorf(p.cur.Span, "attribute @%s must be followed by 'fn', got %s", attrs[0].Name, p.cur.Kind)
	}
	return nil, diag.Errorf(p.cur.Span, "expected item (fn), got %s", p.cur.Kind)
}

// parseAttrs reads zero or more leading attributes. Two surface forms are
// recognised:
//
//	@kernel           — bare name, no args
//	@grad(reverse)    — name with parenthesised expression args
//
// The lexer emits an attached `@<ident>` (no whitespace) as a single
// TkAttr token; bare `@` (matmul) lexes as TkAt and is not consumed
// here. The legacy TkKernel token (a fast-path lex of `@kernel`) is
// normalised to an Attr{Name:"kernel"} for uniformity.
func (p *parser) parseAttrs() ([]ast.Attr, error) {
	var attrs []ast.Attr
	for {
		if p.cur.Kind == lex.TkKernel {
			span := p.cur.Span
			if err := p.bump(); err != nil {
				return nil, err
			}
			attrs = append(attrs, ast.Attr{Name: "kernel", SpanRng: span})
			continue
		}
		if p.cur.Kind != lex.TkAttr {
			return attrs, nil
		}
		startSpan := p.cur.Span
		name := p.cur.Lit
		if err := p.bump(); err != nil {
			return nil, err
		}
		attr := ast.Attr{
			Name:    name,
			SpanRng: startSpan,
		}
		// Optional argument list: @name(args...)
		if p.cur.Kind == lex.TkLParen {
			if err := p.bump(); err != nil {
				return nil, err
			}
			for p.cur.Kind != lex.TkRParen {
				arg, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				attr.Args = append(attr.Args, arg)
				if p.cur.Kind == lex.TkComma {
					if err := p.bump(); err != nil {
						return nil, err
					}
				} else {
					break
				}
			}
			endTok, err := p.expect(lex.TkRParen)
			if err != nil {
				return nil, err
			}
			attr.SpanRng = diag.Span{Start: startSpan.Start, End: endTok.Span.End}
		}
		attrs = append(attrs, attr)
	}
}

func (p *parser) parseFn() (*ast.FnDecl, error) {
	start := p.cur.Span
	if _, err := p.expect(lex.TkFn); err != nil {
		return nil, err
	}
	nameTok, err := p.expect(lex.TkIdent)
	if err != nil {
		return nil, err
	}

	// Optional shape parameters: fn foo[N, M: nat](...)
	var shapeParams []ast.ShapeParam
	if p.cur.Kind == lex.TkLBracket {
		if err := p.bump(); err != nil {
			return nil, err
		}
		for p.cur.Kind != lex.TkRBracket {
			pname, err := p.expect(lex.TkIdent)
			if err != nil {
				return nil, err
			}
			// Optional `: nat` (ignored for now, all shape params are nat)
			if p.cur.Kind == lex.TkColon {
				if err := p.bump(); err != nil {
					return nil, err
				}
				if _, err := p.expect(lex.TkIdent); err != nil { // expect "nat"
					return nil, err
				}
			}
			shapeParams = append(shapeParams, ast.ShapeParam{
				Name: pname.Lit,
				Span: pname.Span,
			})
			if p.cur.Kind == lex.TkComma {
				if err := p.bump(); err != nil {
					return nil, err
				}
			} else {
				break
			}
		}
		if _, err := p.expect(lex.TkRBracket); err != nil {
			return nil, err
		}
	}

	if _, err := p.expect(lex.TkLParen); err != nil {
		return nil, err
	}
	var params []ast.Param
	for p.cur.Kind != lex.TkRParen {
		pname, err := p.expect(lex.TkIdent)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(lex.TkColon); err != nil {
			return nil, err
		}
		ty, err := p.parseType()
		if err != nil {
			return nil, err
		}
		params = append(params, ast.Param{
			Name: pname.Lit, Type: ty,
			Span: diag.Span{Start: pname.Span.Start, End: ty.Span().End},
		})
		if p.cur.Kind == lex.TkComma {
			if err := p.bump(); err != nil {
				return nil, err
			}
		} else {
			break
		}
	}
	if _, err := p.expect(lex.TkRParen); err != nil {
		return nil, err
	}
	if _, err := p.expect(lex.TkArrow); err != nil {
		return nil, err
	}
	retTy, err := p.parseType()
	if err != nil {
		return nil, err
	}

	var body ast.Expr
	var end diag.Span

	// Two forms: `= expr` or `{ block }`
	if p.cur.Kind == lex.TkEq {
		if err := p.bump(); err != nil {
			return nil, err
		}
		body, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
		end = body.Span()
		if p.cur.Kind == lex.TkSemi {
			end = p.cur.Span
			if err := p.bump(); err != nil {
				return nil, err
			}
		}
	} else if p.cur.Kind == lex.TkLBrace {
		body, err = p.parseBlock()
		if err != nil {
			return nil, err
		}
		end = body.Span()
	} else {
		return nil, diag.Errorf(p.cur.Span, "expected '=' or '{' after return type, got %s", p.cur.Kind)
	}

	return &ast.FnDecl{
		NameTok:     nameTok.Span,
		Name:        nameTok.Lit,
		ShapeParams: shapeParams,
		Params:      params,
		RetType:     retTy,
		Body:        body,
		SpanRng:     diag.Span{Start: start.Start, End: end.End},
	}, nil
}

func (p *parser) parseType() (ast.TypeExpr, error) {
	t, err := p.expect(lex.TkIdent)
	if err != nil {
		return nil, err
	}
	elemType := &ast.NamedType{Name: t.Lit, SpanRng: t.Span}

	// Check for tensor type: T[N, M]
	if p.cur.Kind == lex.TkLBracket {
		if err := p.bump(); err != nil {
			return nil, err
		}
		var shape []ast.ShapeExpr
		for p.cur.Kind != lex.TkRBracket {
			dim, err := p.parseShapeExpr()
			if err != nil {
				return nil, err
			}
			shape = append(shape, dim)
			if p.cur.Kind == lex.TkComma {
				if err := p.bump(); err != nil {
					return nil, err
				}
			} else {
				break
			}
		}
		end, err := p.expect(lex.TkRBracket)
		if err != nil {
			return nil, err
		}
		return &ast.TensorType{
			Elem:    elemType,
			Shape:   shape,
			SpanRng: diag.Span{Start: t.Span.Start, End: end.Span.End},
		}, nil
	}

	return elemType, nil
}

// parseShapeExpr parses a shape dimension expression
func (p *parser) parseShapeExpr() (ast.ShapeExpr, error) {
	return p.parseShapeAdditive()
}

func (p *parser) parseShapeAdditive() (ast.ShapeExpr, error) {
	left, err := p.parseShapeMultiplicative()
	if err != nil {
		return nil, err
	}
	for p.cur.Kind == lex.TkPlus || p.cur.Kind == lex.TkMinus {
		op := p.cur.Lit
		if err := p.bump(); err != nil {
			return nil, err
		}
		right, err := p.parseShapeMultiplicative()
		if err != nil {
			return nil, err
		}
		left = &ast.ShapeBinOp{
			Op:      op,
			L:       left,
			R:       right,
			SpanRng: diag.Span{Start: left.Span().Start, End: right.Span().End},
		}
	}
	return left, nil
}

func (p *parser) parseShapeMultiplicative() (ast.ShapeExpr, error) {
	left, err := p.parseShapePrimary()
	if err != nil {
		return nil, err
	}
	for p.cur.Kind == lex.TkStar || p.cur.Kind == lex.TkSlash {
		op := p.cur.Lit
		if err := p.bump(); err != nil {
			return nil, err
		}
		right, err := p.parseShapePrimary()
		if err != nil {
			return nil, err
		}
		left = &ast.ShapeBinOp{
			Op:      op,
			L:       left,
			R:       right,
			SpanRng: diag.Span{Start: left.Span().Start, End: right.Span().End},
		}
	}
	return left, nil
}

func (p *parser) parseShapePrimary() (ast.ShapeExpr, error) {
	switch p.cur.Kind {
	case lex.TkInt:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		v, suf, err := parseIntLit(t.Lit)
		if err != nil {
			return nil, diag.Errorf(t.Span, "invalid shape literal %q", t.Lit)
		}
		if suf != "" {
			return nil, diag.Errorf(t.Span,
				"shape literal cannot carry a type suffix (%q has _%s)", t.Lit, suf)
		}
		return &ast.ShapeLit{Value: v, SpanRng: t.Span}, nil

	case lex.TkIdent:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.ShapeVar{Name: t.Lit, SpanRng: t.Span}, nil

	case lex.TkLParen:
		if err := p.bump(); err != nil {
			return nil, err
		}
		expr, err := p.parseShapeExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(lex.TkRParen); err != nil {
			return nil, err
		}
		return expr, nil
	}

	return nil, diag.Errorf(p.cur.Span, "expected shape expression, got %s", p.cur.Kind)
}

// parseBlock parses { stmts; expr }
func (p *parser) parseBlock() (*ast.Block, error) {
	start, err := p.expect(lex.TkLBrace)
	if err != nil {
		return nil, err
	}

	var stmts []ast.Stmt
	var result ast.Expr

	for p.cur.Kind != lex.TkRBrace && p.cur.Kind != lex.TkEOF {
		// Check if this is a statement (let, return) or an expression
		if p.cur.Kind == lex.TkLet {
			start, name, ty, init, err := p.parseLetHead()
			if err != nil {
				return nil, err
			}
			// `let x = e in body` inside a block: the let-in expression
			// becomes the block's result.
			if p.cur.Kind == lex.TkIn {
				if err := p.bump(); err != nil {
					return nil, err
				}
				body, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				stmts = append(stmts, &ast.LetStmt{
					Name:    name,
					Type:    ty,
					Init:    init,
					SpanRng: diag.Span{Start: start, End: init.Span().End},
				})
				result = body
				break
			}
			// Statement form: `let x = e ;` (or trailing without `;`).
			end := init.Span()
			if p.cur.Kind == lex.TkSemi {
				end = p.cur.Span
				if err := p.bump(); err != nil {
					return nil, err
				}
			}
			stmts = append(stmts, &ast.LetStmt{
				Name:    name,
				Type:    ty,
				Init:    init,
				SpanRng: diag.Span{Start: start, End: end.End},
			})
		} else if p.cur.Kind == lex.TkReturn {
			stmt, err := p.parseReturnStmt()
			if err != nil {
				return nil, err
			}
			stmts = append(stmts, stmt)
		} else {
			// Expression: could be the final result or an ExprStmt
			expr, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			// If followed by `;`, it's an expression statement
			if p.cur.Kind == lex.TkSemi {
				if err := p.bump(); err != nil {
					return nil, err
				}
				stmts = append(stmts, &ast.ExprStmt{X: expr, SpanRng: expr.Span()})
			} else {
				// No semicolon → this is the final result expression
				result = expr
				break
			}
		}
	}

	end, err := p.expect(lex.TkRBrace)
	if err != nil {
		return nil, err
	}

	return &ast.Block{
		Stmts:   stmts,
		Result:  result,
		SpanRng: diag.Span{Start: start.Span.Start, End: end.Span.End},
	}, nil
}

// parseLetHead consumes `let NAME [: TYPE] = INIT` and returns the
// parts. The caller decides whether the next token marks a statement
// (`;` or end of block) or a `let ... in BODY` expression.
func (p *parser) parseLetHead() (diag.Pos, string, ast.TypeExpr, ast.Expr, error) {
	start, err := p.expect(lex.TkLet)
	if err != nil {
		return diag.Pos{}, "", nil, nil, err
	}
	name, err := p.expect(lex.TkIdent)
	if err != nil {
		return diag.Pos{}, "", nil, nil, err
	}
	var ty ast.TypeExpr
	if p.cur.Kind == lex.TkColon {
		if err := p.bump(); err != nil {
			return diag.Pos{}, "", nil, nil, err
		}
		ty, err = p.parseType()
		if err != nil {
			return diag.Pos{}, "", nil, nil, err
		}
	}
	if _, err := p.expect(lex.TkEq); err != nil {
		return diag.Pos{}, "", nil, nil, err
	}
	init, err := p.parseExpr()
	if err != nil {
		return diag.Pos{}, "", nil, nil, err
	}
	return start.Span.Start, name.Lit, ty, init, nil
}

// parseLetIn parses `let x [: T] = e in body` as a primary expression.
// Desugars to a single-statement Block so the existing type-checker
// and CEIR lowering handle binding/scope without duplication.
func (p *parser) parseLetIn() (ast.Expr, error) {
	start, name, ty, init, err := p.parseLetHead()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect(lex.TkIn); err != nil {
		return nil, err
	}
	body, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &ast.Block{
		Stmts: []ast.Stmt{&ast.LetStmt{
			Name:    name,
			Type:    ty,
			Init:    init,
			SpanRng: diag.Span{Start: start, End: init.Span().End},
		}},
		Result:  body,
		SpanRng: diag.Span{Start: start, End: body.Span().End},
	}, nil
}

func (p *parser) parseReturnStmt() (*ast.ReturnStmt, error) {
	start, err := p.expect(lex.TkReturn)
	if err != nil {
		return nil, err
	}
	var val ast.Expr
	end := start.Span
	// Check for expression (not semicolon or closing brace)
	if p.cur.Kind != lex.TkSemi && p.cur.Kind != lex.TkRBrace && p.cur.Kind != lex.TkEOF {
		val, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
		end = val.Span()
	}
	if p.cur.Kind == lex.TkSemi {
		end = p.cur.Span
		if err := p.bump(); err != nil {
			return nil, err
		}
	}
	return &ast.ReturnStmt{
		Value:   val,
		SpanRng: diag.Span{Start: start.Span.Start, End: end.End},
	}, nil
}

// parseExpr is a Pratt-style expression parser.
func (p *parser) parseExpr() (ast.Expr, error) {
	return p.parseBinary(0)
}

// precedence levels
func precOf(op string) int {
	switch op {
	case "|>":
		return 1 // lowest - pipeline chains left to right
	case "||":
		return 5
	case "&&":
		return 6
	case "==", "!=", "<", "<=", ">", ">=":
		return 10
	case "..", "..=":
		return 15 // ranges; non-associative, between comparison and additive
	case "+", "-", ".+", ".-":
		return 20
	case "*", "/", "%", ".*", "./", ".%":
		return 30
	case "@": // matrix multiply
		return 35
	}
	return -1
}

// rangePrec is the parser precedence for the range operators .. and ..=.
const rangePrec = 15

func (p *parser) parseBinary(minPrec int) (ast.Expr, error) {
	lhs, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for {
		// Range operators are non-associative: parse `a..b` and refuse
		// to chain into `a..b..c`. Their precedence is 15, between
		// comparison (10) and additive (20).
		if p.cur.Kind == lex.TkDotDot || p.cur.Kind == lex.TkDotDotEq {
			if rangePrec < minPrec {
				return lhs, nil
			}
			inclusive := p.cur.Kind == lex.TkDotDotEq
			if err := p.bump(); err != nil {
				return nil, err
			}
			rhs, err := p.parseBinary(rangePrec + 1)
			if err != nil {
				return nil, err
			}
			lhs = &ast.RangeExpr{
				Lo:        lhs,
				Hi:        rhs,
				Inclusive: inclusive,
				SpanRng:   diag.Span{Start: lhs.Span().Start, End: rhs.Span().End},
			}
			if p.cur.Kind == lex.TkDotDot || p.cur.Kind == lex.TkDotDotEq {
				return nil, diag.Errorf(p.cur.Span,
					"range operator is non-associative; parenthesize to disambiguate")
			}
			continue
		}
		op := tokOpString(p.cur.Kind)
		if op == "" {
			return lhs, nil
		}
		prec := precOf(op)
		if prec < minPrec {
			return lhs, nil
		}
		if err := p.bump(); err != nil {
			return nil, err
		}
		rhs, err := p.parseBinary(prec + 1)
		if err != nil {
			return nil, err
		}
		// Pipeline operator creates a Pipeline node
		if op == "|>" {
			lhs = &ast.Pipeline{
				L: lhs, R: rhs,
				SpanRng: diag.Span{Start: lhs.Span().Start, End: rhs.Span().End},
			}
		} else {
			lhs = &ast.BinOp{
				Op: op, L: lhs, R: rhs,
				SpanRng: diag.Span{Start: lhs.Span().Start, End: rhs.Span().End},
			}
		}
	}
}

func tokOpString(k lex.Kind) string {
	switch k {
	case lex.TkPlus:
		return "+"
	case lex.TkMinus:
		return "-"
	case lex.TkStar:
		return "*"
	case lex.TkSlash:
		return "/"
	case lex.TkPercent:
		return "%"
	case lex.TkEqEq:
		return "=="
	case lex.TkBangEq:
		return "!="
	case lex.TkLt:
		return "<"
	case lex.TkLtEq:
		return "<="
	case lex.TkGt:
		return ">"
	case lex.TkGtEq:
		return ">="
	case lex.TkAmpAmp:
		return "&&"
	case lex.TkPipePipe:
		return "||"
	case lex.TkPipe:
		return "|>"
	case lex.TkAt:
		return "@"
	case lex.TkDotPlus:
		return ".+"
	case lex.TkDotMinus:
		return ".-"
	case lex.TkDotStar:
		return ".*"
	case lex.TkDotSlash:
		return "./"
	case lex.TkDotPercent:
		return ".%"
	}
	return ""
}

func (p *parser) parseUnary() (ast.Expr, error) {
	if p.cur.Kind == lex.TkMinus {
		opSpan := p.cur.Span
		if err := p.bump(); err != nil {
			return nil, err
		}
		x, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &ast.Unary{
			Op: "-", X: x,
			SpanRng: diag.Span{Start: opSpan.Start, End: x.Span().End},
		}, nil
	}
	if p.cur.Kind == lex.TkBang {
		opSpan := p.cur.Span
		if err := p.bump(); err != nil {
			return nil, err
		}
		x, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &ast.Unary{
			Op: "!", X: x,
			SpanRng: diag.Span{Start: opSpan.Start, End: x.Span().End},
		}, nil
	}
	// Reduction operators: +/ -/ */ //
	switch p.cur.Kind {
	case lex.TkPlusSlash, lex.TkMinusSlash, lex.TkStarSlash, lex.TkSlashSlash:
		opSpan := p.cur.Span
		var op string
		switch p.cur.Kind {
		case lex.TkPlusSlash:
			op = "+"
		case lex.TkMinusSlash:
			op = "-"
		case lex.TkStarSlash:
			op = "*"
		case lex.TkSlashSlash:
			op = "/"
		}
		if err := p.bump(); err != nil {
			return nil, err
		}
		x, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &ast.Reduce{
			Op: op, X: x,
			SpanRng: diag.Span{Start: opSpan.Start, End: x.Span().End},
		}, nil
	}
	return p.parsePostfix()
}

// isFoldableExpr returns true if the expression can be used as a fold function.
// This includes:
// - Lambdas (parenthesized): (|a, b| a + b) / tensor
// - Known fold function names: add, mul, max, min
func isFoldableExpr(e ast.Expr) bool {
	switch n := e.(type) {
	case *ast.Lambda:
		return true
	case *ast.Ident:
		// Only specific function names are recognized as fold operators
		switch n.Name {
		case "add", "mul", "max", "min":
			return true
		}
	}
	return false
}

func (p *parser) parsePostfix() (ast.Expr, error) {
	expr, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	// Handle postfix operators: function calls (with optional shape args), type casts
	for {
		if p.cur.Kind == lex.TkLBracket {
			// Shape arguments for function call: f[N, M](args)
			// Only valid if followed by (
			if err := p.bump(); err != nil {
				return nil, err
			}
			var shapeArgs []ast.ShapeExpr
			for p.cur.Kind != lex.TkRBracket {
				shape, err := p.parseShapeExpr()
				if err != nil {
					return nil, err
				}
				shapeArgs = append(shapeArgs, shape)
				if p.cur.Kind == lex.TkComma {
					if err := p.bump(); err != nil {
						return nil, err
					}
				} else {
					break
				}
			}
			if _, err := p.expect(lex.TkRBracket); err != nil {
				return nil, err
			}
			// Must be followed by ( for function call
			if p.cur.Kind != lex.TkLParen {
				return nil, diag.Errorf(p.cur.Span, "expected '(' after shape arguments, got %s", p.cur.Kind)
			}
			if err := p.bump(); err != nil {
				return nil, err
			}
			var args []ast.Expr
			for p.cur.Kind != lex.TkRParen {
				arg, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				args = append(args, arg)
				if p.cur.Kind == lex.TkComma {
					if err := p.bump(); err != nil {
						return nil, err
					}
				} else {
					break
				}
			}
			end, err := p.expect(lex.TkRParen)
			if err != nil {
				return nil, err
			}
			expr = &ast.Call{
				Fn:        expr,
				ShapeArgs: shapeArgs,
				Args:      args,
				SpanRng:   diag.Span{Start: expr.Span().Start, End: end.Span.End},
			}
		} else if p.cur.Kind == lex.TkLParen {
			// Function call without shape arguments
			if err := p.bump(); err != nil {
				return nil, err
			}
			var args []ast.Expr
			for p.cur.Kind != lex.TkRParen {
				arg, err := p.parseExpr()
				if err != nil {
					return nil, err
				}
				args = append(args, arg)
				if p.cur.Kind == lex.TkComma {
					if err := p.bump(); err != nil {
						return nil, err
					}
				} else {
					break
				}
			}
			end, err := p.expect(lex.TkRParen)
			if err != nil {
				return nil, err
			}
			// Check for special builtins
			callSpan := diag.Span{Start: expr.Span().Start, End: end.Span.End}
			if ident, ok := expr.(*ast.Ident); ok {
				switch ident.Name {
				case "iterate":
					// iterate(n, init, f) -> Iterate node
					if len(args) != 3 {
						return nil, diag.Errorf(callSpan,
							"iterate requires exactly 3 arguments: iterate(n, init, f)")
					}
					expr = &ast.Iterate{
						N:       args[0],
						Init:    args[1],
						Fn:      args[2],
						SpanRng: callSpan,
					}
					continue
				case "tabulate":
					// tabulate(n, |i| f(i)) -> Tabulate node
					if len(args) != 2 {
						return nil, diag.Errorf(callSpan,
							"tabulate requires exactly 2 arguments: tabulate(n, fn)")
					}
					expr = &ast.Tabulate{
						N:       args[0],
						Fn:      args[1],
						SpanRng: callSpan,
					}
					continue
				case "scan":
					// scan(init, |a,x| f(a,x), v) -> Scan node
					if len(args) != 3 {
						return nil, diag.Errorf(callSpan,
							"scan requires exactly 3 arguments: scan(init, fn, v)")
					}
					expr = &ast.Scan{
						Init:    args[0],
						Fn:      args[1],
						V:       args[2],
						SpanRng: callSpan,
					}
					continue
				case "iterate_until":
					// iterate_until(init, step, pred, max) -> IterateUntil node
					if len(args) != 4 {
						return nil, diag.Errorf(callSpan,
							"iterate_until requires exactly 4 arguments: iterate_until(init, step, pred, max)")
					}
					expr = &ast.IterateUntil{
						Init:    args[0],
						Step:    args[1],
						Pred:    args[2],
						Max:     args[3],
						SpanRng: callSpan,
					}
					continue
				case "each":
					// each(v, f) -> Each node
					if len(args) != 2 {
						return nil, diag.Errorf(callSpan,
							"each requires exactly 2 arguments: each(v, fn)")
					}
					expr = &ast.Each{
						V:       args[0],
						Fn:      args[1],
						SpanRng: callSpan,
					}
					continue
				}
			}
			expr = &ast.Call{
				Fn:      expr,
				Args:    args,
				SpanRng: callSpan,
			}
		} else if p.cur.Kind == lex.TkAs {
			// Type cast: expr as Type
			if err := p.bump(); err != nil {
				return nil, err
			}
			targetType, err := p.parseType()
			if err != nil {
				return nil, err
			}
			expr = &ast.Cast{
				X:       expr,
				Target:  targetType,
				SpanRng: diag.Span{Start: expr.Span().Start, End: targetType.Span().End},
			}
		} else if p.cur.Kind == lex.TkSlash && isFoldableExpr(expr) {
			// Fold syntax: f/tensor or f/init/tensor
			// Only when the left side is an identifier or lambda (i.e., a function)
			if err := p.bump(); err != nil {
				return nil, err
			}
			// Parse first operand (could be tensor or init value)
			first, err := p.parseUnary()
			if err != nil {
				return nil, err
			}
			// Check if there's another / for init/tensor form
			if p.cur.Kind == lex.TkSlash {
				// f/init/tensor form
				if err := p.bump(); err != nil {
					return nil, err
				}
				tensor, err := p.parseUnary()
				if err != nil {
					return nil, err
				}
				expr = &ast.Fold{
					Fn:      expr,
					Init:    first,
					X:       tensor,
					SpanRng: diag.Span{Start: expr.Span().Start, End: tensor.Span().End},
				}
			} else {
				// f/tensor form (no init)
				expr = &ast.Fold{
					Fn:      expr,
					Init:    nil,
					X:       first,
					SpanRng: diag.Span{Start: expr.Span().Start, End: first.Span().End},
				}
			}
		} else {
			break
		}
	}
	return expr, nil
}

func (p *parser) parsePrimary() (ast.Expr, error) {
	switch p.cur.Kind {
	case lex.TkInt:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		v, suf, err := parseIntLit(t.Lit)
		if err != nil {
			return nil, diag.Errorf(t.Span, "invalid integer literal %q", t.Lit)
		}
		return &ast.IntLit{Value: v, TypeSuffix: suf, SpanRng: t.Span}, nil

	case lex.TkChar:
		// Char literals are i32-typed integer constants holding the
		// codepoint. The lexer stores the decoded codepoint as a decimal
		// string in Lit so we can reuse strconv here.
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		v, err := strconv.ParseInt(t.Lit, 10, 64)
		if err != nil {
			return nil, diag.Errorf(t.Span, "invalid char literal codepoint %q", t.Lit)
		}
		return &ast.IntLit{Value: v, SpanRng: t.Span}, nil

	case lex.TkString:
		// v0.12 introduces a fat-pointer string type. The lexer has
		// already unescaped the literal into t.Lit; treat that as the
		// canonical UTF-8 bytes the StringLit holds.
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.StringLit{Value: t.Lit, SpanRng: t.Span}, nil

	case lex.TkFloat:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		v, width, err := parseFloatLit(t.Lit)
		if err != nil {
			return nil, diag.Errorf(t.Span, "invalid float literal %q", t.Lit)
		}
		return &ast.FloatLit{Value: v, Width: width, SpanRng: t.Span}, nil

	case lex.TkTrue:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.BoolLit{Value: true, SpanRng: t.Span}, nil

	case lex.TkFalse:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.BoolLit{Value: false, SpanRng: t.Span}, nil

	case lex.TkIdent:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.Ident{Name: t.Lit, SpanRng: t.Span}, nil

	case lex.TkLParen:
		if err := p.bump(); err != nil {
			return nil, err
		}
		e, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(lex.TkRParen); err != nil {
			return nil, err
		}
		return e, nil

	case lex.TkLBrace:
		return p.parseBlock()

	case lex.TkLet:
		return p.parseLetIn()

	case lex.TkIf:
		return p.parseIf()

	case lex.TkMatch:
		return p.parseMatch()

	case lex.TkLBracket:
		return p.parseTensorLit()

	case lex.TkBar:
		return p.parseLambda()
	}
	return nil, diag.Errorf(p.cur.Span, "expected expression, got %s", p.cur.Kind)
}

func (p *parser) parseIf() (*ast.If, error) {
	start, err := p.expect(lex.TkIf)
	if err != nil {
		return nil, err
	}
	cond, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	thenBranch, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	var elseBranch ast.Expr
	end := thenBranch.Span()
	if p.cur.Kind == lex.TkElse {
		if err := p.bump(); err != nil {
			return nil, err
		}
		// else if or else { }
		if p.cur.Kind == lex.TkIf {
			elseBranch, err = p.parseIf()
		} else {
			elseBranch, err = p.parseBlock()
		}
		if err != nil {
			return nil, err
		}
		end = elseBranch.Span()
	}
	return &ast.If{
		Cond:    cond,
		Then:    thenBranch,
		Else:    elseBranch,
		SpanRng: diag.Span{Start: start.Span.Start, End: end.End},
	}, nil
}

func (p *parser) parseMatch() (*ast.Match, error) {
	start, err := p.expect(lex.TkMatch)
	if err != nil {
		return nil, err
	}

	// Parse scrutinee expression
	scrutinee, err := p.parseExpr()
	if err != nil {
		return nil, err
	}

	// Expect opening brace
	if _, err := p.expect(lex.TkLBrace); err != nil {
		return nil, err
	}

	// Parse arms
	var arms []ast.MatchArm
	for p.cur.Kind != lex.TkRBrace && p.cur.Kind != lex.TkEOF {
		arm, err := p.parseMatchArm()
		if err != nil {
			return nil, err
		}
		arms = append(arms, arm)

		// Optional comma between arms
		if p.cur.Kind == lex.TkComma {
			if err := p.bump(); err != nil {
				return nil, err
			}
		}
	}

	end, err := p.expect(lex.TkRBrace)
	if err != nil {
		return nil, err
	}

	return &ast.Match{
		Scrutinee: scrutinee,
		Arms:      arms,
		SpanRng:   diag.Span{Start: start.Span.Start, End: end.Span.End},
	}, nil
}

func (p *parser) parseMatchArm() (ast.MatchArm, error) {
	patStart := p.cur.Span

	// Parse pattern
	pat, err := p.parsePattern()
	if err != nil {
		return ast.MatchArm{}, err
	}

	// Optional guard: `if cond`
	var guard ast.Expr
	if p.cur.Kind == lex.TkIf {
		if err := p.bump(); err != nil {
			return ast.MatchArm{}, err
		}
		guard, err = p.parseExpr()
		if err != nil {
			return ast.MatchArm{}, err
		}
	}

	// Expect =>
	if _, err := p.expect(lex.TkFatArrow); err != nil {
		return ast.MatchArm{}, err
	}

	// Parse body expression
	body, err := p.parseExpr()
	if err != nil {
		return ast.MatchArm{}, err
	}

	return ast.MatchArm{
		Pattern: pat,
		Guard:   guard,
		Body:    body,
		SpanRng: diag.Span{Start: patStart.Start, End: body.Span().End},
	}, nil
}

func (p *parser) parseTensorLit() (*ast.TensorLit, error) {
	start, err := p.expect(lex.TkLBracket)
	if err != nil {
		return nil, err
	}

	var elems []ast.Expr
	for p.cur.Kind != lex.TkRBracket && p.cur.Kind != lex.TkEOF {
		elem, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		elems = append(elems, elem)
		if p.cur.Kind == lex.TkComma {
			if err := p.bump(); err != nil {
				return nil, err
			}
		} else {
			break
		}
	}

	end, err := p.expect(lex.TkRBracket)
	if err != nil {
		return nil, err
	}

	return &ast.TensorLit{
		Elems:   elems,
		SpanRng: diag.Span{Start: start.Span.Start, End: end.Span.End},
	}, nil
}

// parseLambda parses an anonymous function: |x, y| body or |x: T, y: T| body
func (p *parser) parseLambda() (*ast.Lambda, error) {
	start, err := p.expect(lex.TkBar)
	if err != nil {
		return nil, err
	}

	var params []ast.LambdaParam
	for p.cur.Kind != lex.TkBar && p.cur.Kind != lex.TkEOF {
		nameTok, err := p.expect(lex.TkIdent)
		if err != nil {
			return nil, err
		}

		var ty ast.TypeExpr
		if p.cur.Kind == lex.TkColon {
			if err := p.bump(); err != nil {
				return nil, err
			}
			ty, err = p.parseType()
			if err != nil {
				return nil, err
			}
		}

		paramSpan := nameTok.Span
		if ty != nil {
			paramSpan = diag.Span{Start: nameTok.Span.Start, End: ty.Span().End}
		}
		params = append(params, ast.LambdaParam{
			Name: nameTok.Lit,
			Type: ty,
			Span: paramSpan,
		})

		if p.cur.Kind == lex.TkComma {
			if err := p.bump(); err != nil {
				return nil, err
			}
		} else {
			break
		}
	}

	if _, err := p.expect(lex.TkBar); err != nil {
		return nil, err
	}

	body, err := p.parseExpr()
	if err != nil {
		return nil, err
	}

	return &ast.Lambda{
		Params:  params,
		Body:    body,
		SpanRng: diag.Span{Start: start.Span.Start, End: body.Span().End},
	}, nil
}

func (p *parser) parsePattern() (ast.Pattern, error) {
	switch p.cur.Kind {
	case lex.TkInt:
		// Integer literal pattern
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		v, suf, err := parseIntLit(t.Lit)
		if err != nil {
			return nil, diag.Errorf(t.Span, "invalid integer literal %q", t.Lit)
		}
		return &ast.LitPat{
			Value:   &ast.IntLit{Value: v, TypeSuffix: suf, SpanRng: t.Span},
			SpanRng: t.Span,
		}, nil

	case lex.TkTrue:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.LitPat{
			Value:   &ast.BoolLit{Value: true, SpanRng: t.Span},
			SpanRng: t.Span,
		}, nil

	case lex.TkFalse:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		return &ast.LitPat{
			Value:   &ast.BoolLit{Value: false, SpanRng: t.Span},
			SpanRng: t.Span,
		}, nil

	case lex.TkIdent:
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		// `_` is wildcard, other identifiers are binding patterns
		if t.Lit == "_" {
			return &ast.WildcardPat{SpanRng: t.Span}, nil
		}
		return &ast.BindPat{Name: t.Lit, SpanRng: t.Span}, nil

	case lex.TkMinus:
		// Negative integer literal pattern
		start := p.cur.Span
		if err := p.bump(); err != nil {
			return nil, err
		}
		if p.cur.Kind != lex.TkInt {
			return nil, diag.Errorf(p.cur.Span, "expected integer after '-' in pattern")
		}
		t := p.cur
		if err := p.bump(); err != nil {
			return nil, err
		}
		v, suf, err := parseIntLit(t.Lit)
		if err != nil {
			return nil, diag.Errorf(t.Span, "invalid integer literal %q", t.Lit)
		}
		return &ast.LitPat{
			Value:   &ast.IntLit{Value: -v, TypeSuffix: suf, SpanRng: diag.Span{Start: start.Start, End: t.Span.End}},
			SpanRng: diag.Span{Start: start.Start, End: t.Span.End},
		}, nil
	}

	return nil, diag.Errorf(p.cur.Span, "expected pattern, got %s", p.cur.Kind)
}
