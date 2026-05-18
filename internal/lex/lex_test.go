package lex

import (
	"strings"
	"testing"
)

// tokens lexes the given source and returns the kind sequence (excluding EOF).
// Errors are reported through the test.
func tokens(t *testing.T, src string) []Token {
	t.Helper()
	l := New("test.esq", []byte(src))
	var out []Token
	for {
		tk, err := l.Next()
		if err != nil {
			t.Fatalf("lex error: %v", err)
		}
		if tk.Kind == TkEOF {
			break
		}
		out = append(out, tk)
	}
	return out
}

func kinds(toks []Token) []Kind {
	out := make([]Kind, len(toks))
	for i, t := range toks {
		out[i] = t.Kind
	}
	return out
}

func TestLexBasic(t *testing.T) {
	toks := tokens(t, "fn main() = 42")
	got := kinds(toks)
	want := []Kind{TkFn, TkIdent, TkLParen, TkRParen, TkEq, TkInt}
	if len(got) != len(want) {
		t.Fatalf("len got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
	if toks[1].Lit != "main" {
		t.Errorf("ident lit = %q", toks[1].Lit)
	}
	if toks[5].Lit != "42" {
		t.Errorf("int lit = %q", toks[5].Lit)
	}
}

// TestLexRangeAndIn verifies that `in` is still a keyword while
// `for`/`while`/`break`/`continue` lex as plain identifiers (they were
// removed from the keyword set in v0.10).
func TestLexRangeAndIn(t *testing.T) {
	toks := tokens(t, "for i in 0..10 { while x { break; continue } }")
	got := kinds(toks)
	want := []Kind{
		TkIdent, TkIdent, TkIn, TkInt, TkDotDot, TkInt, TkLBrace,
		TkIdent, TkIdent, TkLBrace, TkIdent, TkSemi, TkIdent, TkRBrace,
		TkRBrace,
	}
	if len(got) != len(want) {
		t.Fatalf("kinds len mismatch: got %d want %d:\n got %v\n want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
	// Sanity check the identifier literals.
	wantLits := map[int]string{0: "for", 7: "while", 10: "break", 12: "continue"}
	for i, lit := range wantLits {
		if toks[i].Lit != lit {
			t.Errorf("toks[%d].Lit = %q, want %q", i, toks[i].Lit, lit)
		}
	}
}

// TestLexRangeOps verifies that `..` and `..=` are tokenized correctly.
func TestLexRangeOps(t *testing.T) {
	toks := tokens(t, "0..5 1..=10")
	got := kinds(toks)
	want := []Kind{TkInt, TkDotDot, TkInt, TkInt, TkDotDotEq, TkInt}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
}

func TestLexString(t *testing.T) {
	toks := tokens(t, `"hello\nworld"`)
	if len(toks) != 1 || toks[0].Kind != TkString {
		t.Fatalf("kinds = %v", kinds(toks))
	}
	if toks[0].Lit != "hello\nworld" {
		t.Errorf("string lit = %q", toks[0].Lit)
	}
}

func TestLexStringWithEscapes(t *testing.T) {
	toks := tokens(t, `"a\\b\"c\t"`)
	want := "a\\b\"c\t"
	if len(toks) != 1 || toks[0].Kind != TkString {
		t.Fatalf("kinds = %v", kinds(toks))
	}
	if toks[0].Lit != want {
		t.Errorf("got %q want %q", toks[0].Lit, want)
	}
}

func TestLexUnterminatedString(t *testing.T) {
	l := New("t.esq", []byte(`"unterminated`))
	_, err := l.Next()
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "unterminated") {
		t.Errorf("err = %v", err)
	}
}

func TestLexChar(t *testing.T) {
	toks := tokens(t, `'a'`)
	if len(toks) != 1 || toks[0].Kind != TkChar {
		t.Fatalf("kinds = %v", kinds(toks))
	}
	if toks[0].Lit != "97" {
		t.Errorf("char lit = %q want %q", toks[0].Lit, "97")
	}
}

func TestLexCharEscape(t *testing.T) {
	toks := tokens(t, `'\n'`)
	if len(toks) != 1 || toks[0].Kind != TkChar {
		t.Fatalf("kinds = %v", kinds(toks))
	}
	if toks[0].Lit != "10" {
		t.Errorf("char lit = %q", toks[0].Lit)
	}
}

func TestLexFloatLiterals(t *testing.T) {
	toks := tokens(t, "3.14 1.0_f64 1e5 2.5e-3")
	got := kinds(toks)
	want := []Kind{TkFloat, TkFloat, TkFloat, TkFloat}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] %v", i, got[i])
		}
	}
}

func TestLexIntTypeSuffix(t *testing.T) {
	// _i64 should leave the token an int (not a float).
	toks := tokens(t, "42_i64 7_u8 100_i32")
	for i, tk := range toks {
		if tk.Kind != TkInt {
			t.Errorf("toks[%d].Kind = %v want TkInt (lit=%q)", i, tk.Kind, tk.Lit)
		}
	}
}

func TestLexTensorOps(t *testing.T) {
	toks := tokens(t, "a .+ b .- c .* d ./ e +/ f -/ g */ h // i @ j |> k")
	got := kinds(toks)
	want := []Kind{
		TkIdent, TkDotPlus, TkIdent, TkDotMinus, TkIdent, TkDotStar, TkIdent,
		TkDotSlash, TkIdent, TkPlusSlash, TkIdent, TkMinusSlash, TkIdent,
		TkStarSlash, TkIdent, TkSlashSlash, TkIdent, TkAt, TkIdent, TkPipe,
		TkIdent,
	}
	if len(got) != len(want) {
		t.Fatalf("got %v\nwant %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %v want %v", i, got[i], want[i])
		}
	}
}

func TestLexComments(t *testing.T) {
	toks := tokens(t, "# comment\nfn /* nested /* block */ */ x")
	got := kinds(toks)
	want := []Kind{TkFn, TkIdent}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

// TestLexReduceVsComment locks in the v0.14 disambiguation: `//`
// produces TkSlashSlash (reduction operator), and `#` is the only
// line-comment lead-in.
func TestLexReduceVsComment(t *testing.T) {
	// `//` is the reduction operator; the trailing `# tail` is a comment.
	toks := tokens(t, "//xs  # tail")
	got := kinds(toks)
	want := []Kind{TkSlashSlash, TkIdent}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestLexSpans(t *testing.T) {
	toks := tokens(t, "fn x")
	if toks[0].Span.Start.Line != 1 || toks[0].Span.Start.Col != 1 {
		t.Errorf("'fn' span start = %v", toks[0].Span.Start)
	}
	if toks[1].Span.Start.Col != 4 {
		t.Errorf("'x' span start col = %d want 4", toks[1].Span.Start.Col)
	}
}

func TestLexKernelAttribute(t *testing.T) {
	toks := tokens(t, "@kernel fn k() = 0")
	got := kinds(toks)
	want := []Kind{TkKernel, TkFn, TkIdent, TkLParen, TkRParen, TkEq, TkInt}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
