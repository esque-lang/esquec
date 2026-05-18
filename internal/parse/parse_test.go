package parse

import (
	"strings"
	"testing"

	"github.com/esque-lang/esquec/internal/ast"
)

func parseStr(t *testing.T, src string) *ast.File {
	t.Helper()
	f, err := File("test.esq", []byte(src))
	if err != nil {
		t.Fatalf("parse error: %v\nsrc:\n%s", err, src)
	}
	return f
}

func mustFail(t *testing.T, src, wantSubstr string) {
	t.Helper()
	_, err := File("test.esq", []byte(src))
	if err == nil {
		t.Fatalf("expected error containing %q, got success\nsrc:\n%s", wantSubstr, src)
	}
	if !strings.Contains(err.Error(), wantSubstr) {
		t.Fatalf("error %q does not contain %q", err.Error(), wantSubstr)
	}
}

func TestParseFnSingleExpr(t *testing.T) {
	f := parseStr(t, `fn main() -> i32 = 42`)
	if len(f.Items) != 1 {
		t.Fatalf("items = %d", len(f.Items))
	}
	fn, ok := f.Items[0].(*ast.FnDecl)
	if !ok {
		t.Fatalf("not FnDecl: %T", f.Items[0])
	}
	if fn.Name != "main" {
		t.Errorf("name = %q", fn.Name)
	}
	if len(fn.Params) != 0 {
		t.Errorf("params = %d", len(fn.Params))
	}
	lit, ok := fn.Body.(*ast.IntLit)
	if !ok {
		t.Fatalf("body not IntLit: %T", fn.Body)
	}
	if lit.Value != 42 {
		t.Errorf("value = %d", lit.Value)
	}
}

func TestParseFnBlockBody(t *testing.T) {
	f := parseStr(t, `fn main() -> i32 { let x = 1; let y = 2; x + y }`)
	fn := f.Items[0].(*ast.FnDecl)
	blk, ok := fn.Body.(*ast.Block)
	if !ok {
		t.Fatalf("body not Block: %T", fn.Body)
	}
	if len(blk.Stmts) != 2 {
		t.Errorf("stmts = %d", len(blk.Stmts))
	}
	if blk.Result == nil {
		t.Error("missing trailing expression")
	}
}

func TestParseShapeParams(t *testing.T) {
	f := parseStr(t, `fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)`)
	fn := f.Items[0].(*ast.FnDecl)
	if len(fn.ShapeParams) != 1 {
		t.Fatalf("shape params = %d", len(fn.ShapeParams))
	}
	if fn.ShapeParams[0].Name != "N" {
		t.Errorf("shape param name = %q", fn.ShapeParams[0].Name)
	}
	if len(fn.Params) != 2 {
		t.Fatalf("params = %d", len(fn.Params))
	}
}

func TestParseTensorType2D(t *testing.T) {
	f := parseStr(t, `fn id(x: f32[3, 4]) -> f32[3, 4] = x`)
	fn := f.Items[0].(*ast.FnDecl)
	tt, ok := fn.Params[0].Type.(*ast.TensorType)
	if !ok {
		t.Fatalf("not TensorType: %T", fn.Params[0].Type)
	}
	if len(tt.Shape) != 2 {
		t.Errorf("shape len = %d", len(tt.Shape))
	}
}

func TestParseShapeArithmetic(t *testing.T) {
	f := parseStr(t, `fn ext[N](x: f32[N+1]) -> f32[N+1] = x`)
	fn := f.Items[0].(*ast.FnDecl)
	tt := fn.Params[0].Type.(*ast.TensorType)
	if _, ok := tt.Shape[0].(*ast.ShapeBinOp); !ok {
		t.Fatalf("shape[0] not ShapeBinOp: %T", tt.Shape[0])
	}
}

func TestParseTensorLit(t *testing.T) {
	f := parseStr(t, `fn k() -> f32[3] = [1.0, 2.0, 3.0]`)
	fn := f.Items[0].(*ast.FnDecl)
	tl, ok := fn.Body.(*ast.TensorLit)
	if !ok {
		t.Fatalf("not TensorLit: %T", fn.Body)
	}
	if len(tl.Elems) != 3 {
		t.Errorf("elems = %d", len(tl.Elems))
	}
}

func TestParseReduce(t *testing.T) {
	f := parseStr(t, `fn s(x: f32[3]) -> f32 = +/x`)
	fn := f.Items[0].(*ast.FnDecl)
	if _, ok := fn.Body.(*ast.Reduce); !ok {
		t.Fatalf("body = %T", fn.Body)
	}
}

func TestParsePipeline(t *testing.T) {
	f := parseStr(t, `fn s(x: f32[3]) -> f32[3] = x |> id`)
	fn := f.Items[0].(*ast.FnDecl)
	if _, ok := fn.Body.(*ast.Pipeline); !ok {
		t.Fatalf("body = %T", fn.Body)
	}
}

func TestParseIfExpr(t *testing.T) {
	f := parseStr(t, `fn s(x: i32) -> i32 = if x > 0 { 1 } else { -1 }`)
	fn := f.Items[0].(*ast.FnDecl)
	if _, ok := fn.Body.(*ast.If); !ok {
		t.Fatalf("body = %T", fn.Body)
	}
}

func TestParseBinaryPrecedence(t *testing.T) {
	// 1 + 2 * 3 should parse as 1 + (2 * 3): top-level op is +.
	f := parseStr(t, `fn k() -> i32 = 1 + 2 * 3`)
	fn := f.Items[0].(*ast.FnDecl)
	bo, ok := fn.Body.(*ast.BinOp)
	if !ok {
		t.Fatalf("body = %T", fn.Body)
	}
	if bo.Op != "+" {
		t.Errorf("top op = %q want +", bo.Op)
	}
	rhs, ok := bo.R.(*ast.BinOp)
	if !ok {
		t.Fatalf("rhs = %T", bo.R)
	}
	if rhs.Op != "*" {
		t.Errorf("rhs op = %q want *", rhs.Op)
	}
}

func TestParseUnclosedBrace(t *testing.T) {
	mustFail(t, `fn main() -> i32 { 1`, "")
}

func TestParseNoTopLevelStmt(t *testing.T) {
	mustFail(t, `let x = 1`, "")
}

func TestParseUnknownToken(t *testing.T) {
	mustFail(t, `fn main() -> i32 = ?`, "")
}

// TestParseRecoverMultipleItemErrors verifies that two independent broken
// functions both produce diagnostics rather than the parser stopping at
// the first one.
func TestParseRecoverMultipleItemErrors(t *testing.T) {
	src := `fn a() -> i32 = ?
fn b() -> i32 = ?
fn c() -> i32 = 3`
	f, bag := FileWithBag("test.esq", []byte(src))
	if !bag.HasErrors() {
		t.Fatalf("expected errors, got none")
	}
	if got := len(bag.Diags); got < 2 {
		t.Fatalf("want >=2 diagnostics, got %d: %v", got, bag.Diags)
	}
	// Recovery should let `fn c` parse successfully.
	foundC := false
	for _, it := range f.Items {
		if fn, ok := it.(*ast.FnDecl); ok && fn.Name == "c" {
			foundC = true
		}
	}
	if !foundC {
		t.Errorf("recovery did not produce fn c; items=%v", itemNames(f))
	}
}

// TestParseRecoverAfterBadHeader verifies the parser recovers from a
// malformed header (missing return type) and still parses the next fn.
func TestParseRecoverAfterBadHeader(t *testing.T) {
	src := `fn bad() = 1
fn good() -> i32 = 2`
	f, bag := FileWithBag("test.esq", []byte(src))
	if !bag.HasErrors() {
		t.Fatal("expected error from missing -> on fn bad")
	}
	foundGood := false
	for _, it := range f.Items {
		if fn, ok := it.(*ast.FnDecl); ok && fn.Name == "good" {
			foundGood = true
		}
	}
	if !foundGood {
		t.Errorf("recovery did not produce fn good; items=%v", itemNames(f))
	}
}

// TestParseFileFirstErrorAPI verifies that File still surfaces the first
// diagnostic for backwards compatibility with single-error consumers.
func TestParseFileFirstErrorAPI(t *testing.T) {
	src := `fn a() -> i32 = ?
fn b() -> i32 = ?`
	_, err := File("test.esq", []byte(src))
	if err == nil {
		t.Fatal("expected error")
	}
}

// TestParseLegacyKernelAttr verifies the legacy `@kernel` shortcut still
// produces an Attr with Name == "kernel" and sets IsKernel on the fn.
func TestParseLegacyKernelAttr(t *testing.T) {
	f := parseStr(t, `@kernel fn k() -> i32 = 0`)
	fn := f.Items[0].(*ast.FnDecl)
	if !fn.IsKernel {
		t.Errorf("IsKernel=false, want true")
	}
	if len(fn.Attrs) != 1 {
		t.Fatalf("Attrs len=%d, want 1", len(fn.Attrs))
	}
	if fn.Attrs[0].Name != "kernel" {
		t.Errorf("attr name=%q, want kernel", fn.Attrs[0].Name)
	}
	if len(fn.Attrs[0].Args) != 0 {
		t.Errorf("attr args=%d, want 0", len(fn.Attrs[0].Args))
	}
}

// TestParseGenericAttrNoArgs verifies a non-kernel attribute parses and
// is recorded.
func TestParseGenericAttrNoArgs(t *testing.T) {
	f := parseStr(t, `@inline fn k() -> i32 = 0`)
	fn := f.Items[0].(*ast.FnDecl)
	if fn.IsKernel {
		t.Errorf("IsKernel=true on @inline-only fn")
	}
	if len(fn.Attrs) != 1 || fn.Attrs[0].Name != "inline" {
		t.Errorf("Attrs=%+v", fn.Attrs)
	}
}

// TestParseGenericAttrWithArgs verifies @name(args) parses with each
// arg being an Expr.
func TestParseGenericAttrWithArgs(t *testing.T) {
	f := parseStr(t, `@inline(1, 2) fn k() -> i32 = 0`)
	fn := f.Items[0].(*ast.FnDecl)
	if len(fn.Attrs) != 1 {
		t.Fatalf("Attrs len=%d", len(fn.Attrs))
	}
	a := fn.Attrs[0]
	if a.Name != "inline" {
		t.Errorf("name=%q", a.Name)
	}
	if len(a.Args) != 2 {
		t.Fatalf("args len=%d", len(a.Args))
	}
	if _, ok := a.Args[0].(*ast.IntLit); !ok {
		t.Errorf("arg0 type=%T", a.Args[0])
	}
}

// TestParseMultipleAttrs verifies a stack of attributes is preserved in
// declaration order.
func TestParseMultipleAttrs(t *testing.T) {
	f := parseStr(t, `@kernel @inline(always) fn k() -> i32 = 0`)
	fn := f.Items[0].(*ast.FnDecl)
	if !fn.IsKernel {
		t.Errorf("IsKernel=false")
	}
	if len(fn.Attrs) != 2 {
		t.Fatalf("Attrs len=%d, want 2", len(fn.Attrs))
	}
	if fn.Attrs[0].Name != "kernel" || fn.Attrs[1].Name != "inline" {
		t.Errorf("attr names=[%s, %s]", fn.Attrs[0].Name, fn.Attrs[1].Name)
	}
}

// TestParseRangeBasic verifies `lo..hi` parses to a RangeExpr with
// inclusive=false.
func TestParseRangeBasic(t *testing.T) {
	f := parseStr(t, `fn k() -> i32 = +/(0..5)`)
	fn := f.Items[0].(*ast.FnDecl)
	red, ok := fn.Body.(*ast.Reduce)
	if !ok {
		t.Fatalf("body not Reduce: %T", fn.Body)
	}
	rng, ok := red.X.(*ast.RangeExpr)
	if !ok {
		t.Fatalf("reduce arg not RangeExpr: %T", red.X)
	}
	if rng.Inclusive {
		t.Errorf("inclusive=true, want false for `..`")
	}
	if rng.Lo.(*ast.IntLit).Value != 0 || rng.Hi.(*ast.IntLit).Value != 5 {
		t.Errorf("range bounds wrong: %+v..%+v", rng.Lo, rng.Hi)
	}
}

// TestParseRangeInclusive verifies `..=` produces Inclusive=true.
func TestParseRangeInclusive(t *testing.T) {
	f := parseStr(t, `fn k() -> i32 = +/(1..=5)`)
	fn := f.Items[0].(*ast.FnDecl)
	rng := fn.Body.(*ast.Reduce).X.(*ast.RangeExpr)
	if !rng.Inclusive {
		t.Errorf("inclusive=false, want true for `..=`")
	}
}

// TestParseRangePrecedenceVsArith confirms `0..N+1` parses as `0..(N+1)`
// because `+` (precedence 16) binds tighter than `..` (precedence 15).
func TestParseRangePrecedenceVsArith(t *testing.T) {
	f := parseStr(t, `fn k(n: i32) -> i32 = +/(0..n+1)`)
	fn := f.Items[0].(*ast.FnDecl)
	rng := fn.Body.(*ast.Reduce).X.(*ast.RangeExpr)
	if _, ok := rng.Hi.(*ast.BinOp); !ok {
		t.Fatalf("Hi not BinOp: %T", rng.Hi)
	}
}

// TestParseRangePrecedenceVsPipeline confirms `0..5 |> f` groups as
// `(0..5) |> f` because `|>` (precedence 5) is weaker than `..` (15).
func TestParseRangePrecedenceVsPipeline(t *testing.T) {
	f := parseStr(t, `
fn id(x: i32[5]) -> i32[5] = x
fn k() -> i32[5] = 0..5 |> id`)
	fn := f.Items[1].(*ast.FnDecl)
	pipe, ok := fn.Body.(*ast.Pipeline)
	if !ok {
		t.Fatalf("body not Pipeline: %T", fn.Body)
	}
	if _, ok := pipe.L.(*ast.RangeExpr); !ok {
		t.Errorf("pipeline LHS not RangeExpr: %T", pipe.L)
	}
}

// TestParseRangeNonAssociative rejects `a..b..c` because `..` is
// non-associative.
func TestParseRangeNonAssociative(t *testing.T) {
	mustFail(t, `fn k() -> i32 = +/(0..3..5)`, "")
}

// TestParseTabulate verifies tabulate(N, lambda) lowers to a Tabulate
// node with the right pieces.
func TestParseTabulate(t *testing.T) {
	f := parseStr(t, `fn k() -> i32 = +/(tabulate(5, |i| i*i))`)
	fn := f.Items[0].(*ast.FnDecl)
	tb, ok := fn.Body.(*ast.Reduce).X.(*ast.Tabulate)
	if !ok {
		t.Fatalf("body inner not Tabulate: %T", fn.Body.(*ast.Reduce).X)
	}
	if tb.N.(*ast.IntLit).Value != 5 {
		t.Errorf("N = %v", tb.N)
	}
	if _, ok := tb.Fn.(*ast.Lambda); !ok {
		t.Errorf("Fn not Lambda: %T", tb.Fn)
	}
}

// TestParseScan verifies scan(init, fn, v) parses into the expected
// AST shape.
func TestParseScan(t *testing.T) {
	f := parseStr(t, `fn k() -> i32 = +/(scan(0, |a, x| a+x, [1,2,3]))`)
	fn := f.Items[0].(*ast.FnDecl)
	sc, ok := fn.Body.(*ast.Reduce).X.(*ast.Scan)
	if !ok {
		t.Fatalf("body inner not Scan: %T", fn.Body.(*ast.Reduce).X)
	}
	if sc.Init.(*ast.IntLit).Value != 0 {
		t.Errorf("init = %v", sc.Init)
	}
	if _, ok := sc.Fn.(*ast.Lambda); !ok {
		t.Errorf("Fn not Lambda: %T", sc.Fn)
	}
	if _, ok := sc.V.(*ast.TensorLit); !ok {
		t.Errorf("V not TensorLit: %T", sc.V)
	}
}

// TestParseIterateUntil verifies iterate_until(init, step, pred, max)
// produces an IterateUntil node.
func TestParseIterateUntil(t *testing.T) {
	f := parseStr(t, `fn k() -> i32 = iterate_until(0, |s| s+1, |s| s == 5, 10)`)
	fn := f.Items[0].(*ast.FnDecl)
	iu, ok := fn.Body.(*ast.IterateUntil)
	if !ok {
		t.Fatalf("body not IterateUntil: %T", fn.Body)
	}
	if iu.Init.(*ast.IntLit).Value != 0 {
		t.Errorf("init = %v", iu.Init)
	}
	if iu.Max.(*ast.IntLit).Value != 10 {
		t.Errorf("max = %v", iu.Max)
	}
}

// TestParseEach verifies each(v, fn) produces an Each node.
func TestParseEach(t *testing.T) {
	f := parseStr(t, `fn k() -> i32 = { each(0..3, print_i32); 0 }`)
	fn := f.Items[0].(*ast.FnDecl)
	blk, ok := fn.Body.(*ast.Block)
	if !ok {
		t.Fatalf("body not Block: %T", fn.Body)
	}
	if len(blk.Stmts) != 1 {
		t.Fatalf("stmts = %d, want 1", len(blk.Stmts))
	}
	es, ok := blk.Stmts[0].(*ast.ExprStmt)
	if !ok {
		t.Fatalf("stmt not ExprStmt: %T", blk.Stmts[0])
	}
	ea, ok := es.X.(*ast.Each)
	if !ok {
		t.Fatalf("not Each: %T", es.X)
	}
	if _, ok := ea.V.(*ast.RangeExpr); !ok {
		t.Errorf("V not RangeExpr: %T", ea.V)
	}
	id, ok := ea.Fn.(*ast.Ident)
	if !ok || id.Name != "print_i32" {
		t.Errorf("Fn not Ident(print_i32): %T %v", ea.Fn, ea.Fn)
	}
}

// TestParseTabulateArity rejects tabulate with the wrong number of args.
func TestParseTabulateArity(t *testing.T) {
	mustFail(t, `fn k() -> i32 = +/(tabulate(5))`, "tabulate")
	mustFail(t, `fn k() -> i32 = +/(tabulate(5, |i| i, 99))`, "tabulate")
}

// TestParseScanArity rejects scan with the wrong number of args.
func TestParseScanArity(t *testing.T) {
	mustFail(t, `fn k() -> i32 = +/(scan(0, |a,x| a+x))`, "scan")
}

// TestParseIterateUntilArity rejects iterate_until with the wrong arity.
func TestParseIterateUntilArity(t *testing.T) {
	mustFail(t, `fn k() -> i32 = iterate_until(0, |s| s+1, |s| s == 5)`, "iterate_until")
}

// TestParseEachArity rejects each with the wrong arity.
func TestParseEachArity(t *testing.T) {
	mustFail(t, `fn k() -> i32 = { each(0..3); 0 }`, "each")
}

// TestParseCharLiteral verifies char literals parse to IntLit with the
// codepoint as the value.
func TestParseCharLiteral(t *testing.T) {
	f := parseStr(t, `fn main() -> i32 = 'A'`)
	fn := f.Items[0].(*ast.FnDecl)
	lit, ok := fn.Body.(*ast.IntLit)
	if !ok {
		t.Fatalf("body not IntLit: %T", fn.Body)
	}
	if lit.Value != 65 {
		t.Errorf("value = %d want 65", lit.Value)
	}
}

// TestParseCharLiteralEscape verifies escape sequences in char literals.
func TestParseCharLiteralEscape(t *testing.T) {
	f := parseStr(t, `fn main() -> i32 = '\n'`)
	fn := f.Items[0].(*ast.FnDecl)
	lit := fn.Body.(*ast.IntLit)
	if lit.Value != 10 {
		t.Errorf("value = %d want 10", lit.Value)
	}
}

// TestParseIntSuffix verifies that `_i32`, `_i64`, `_u8` suffixes are
// parsed and stored on the AST.
func TestParseIntSuffix(t *testing.T) {
	cases := []struct {
		src    string
		val    int64
		suffix string
	}{
		{`fn k() -> i32 = 42_i32`, 42, "i32"},
		{`fn k() -> i32 = 42_i64`, 42, "i64"},
		{`fn k() -> i32 = 7_u8`, 7, "u8"},
		{`fn k() -> i32 = 100_u32`, 100, "u32"},
		{`fn k() -> i32 = 5`, 5, ""}, // unsuffixed
	}
	for _, tc := range cases {
		f := parseStr(t, tc.src)
		fn := f.Items[0].(*ast.FnDecl)
		lit, ok := fn.Body.(*ast.IntLit)
		if !ok {
			t.Errorf("%q: body not IntLit: %T", tc.src, fn.Body)
			continue
		}
		if lit.Value != tc.val {
			t.Errorf("%q: value = %d want %d", tc.src, lit.Value, tc.val)
		}
		if lit.TypeSuffix != tc.suffix {
			t.Errorf("%q: suffix = %q want %q", tc.src, lit.TypeSuffix, tc.suffix)
		}
	}
}

// TestParseStringLit verifies the parser accepts a string literal and
// records its (escape-decoded) value on the resulting StringLit node.
func TestParseStringLit(t *testing.T) {
	f := parseStr(t, `fn k() -> string = "hello\nworld"`)
	fn := f.Items[0].(*ast.FnDecl)
	lit, ok := fn.Body.(*ast.StringLit)
	if !ok {
		t.Fatalf("body not StringLit: %T", fn.Body)
	}
	if lit.Value != "hello\nworld" {
		t.Errorf("value = %q, want %q", lit.Value, "hello\nworld")
	}
}

// TestParseLoopKeywordsAsIdents verifies that `for`, `while`, `break`,
// `continue` are now ordinary identifiers (no longer keywords).
func TestParseLoopKeywordsAsIdents(t *testing.T) {
	parseStr(t, `fn k(for: i32, while: i32, break: i32, continue: i32) -> i32 = for + while + break + continue`)
}

func itemNames(f *ast.File) []string {
	var out []string
	for _, it := range f.Items {
		if fn, ok := it.(*ast.FnDecl); ok {
			out = append(out, fn.Name)
		}
	}
	return out
}
