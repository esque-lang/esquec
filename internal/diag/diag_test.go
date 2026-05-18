package diag

import (
	"strings"
	"testing"
)

func mkSpan(file string, sLine, sCol, eLine, eCol int) Span {
	return Span{
		Start: Pos{File: file, Line: sLine, Col: sCol},
		End:   Pos{File: file, Line: eLine, Col: eCol},
	}
}

func TestErrorfRendersHeader(t *testing.T) {
	d := Errorf(mkSpan("a.esq", 2, 5, 2, 8), "expected %s, got %s", "x", "y")
	got := d.Error()
	want := "a.esq:2:5: error: expected x, got y"
	if got != want {
		t.Fatalf("Error() = %q want %q", got, want)
	}
}

func TestRenderSnippet(t *testing.T) {
	src := []byte("fn main() = 1\nfn other() = 2\n")
	d := Errorf(mkSpan("a.esq", 2, 4, 2, 9), "bad function").
		WithNote(mkSpan("a.esq", 1, 4, 1, 8), "first declared here").
		WithHelp("rename one of them")
	out := d.Render(src)
	mustContain := []string{
		"error: bad function",
		"--> a.esq:2:4",
		"   2 | fn other() = 2",
		"^^^^^",
		"note: first declared here",
		"--> a.esq:1:4",
		"   1 | fn main() = 1",
		"help: rename one of them",
	}
	for _, sub := range mustContain {
		if !strings.Contains(out, sub) {
			t.Errorf("render missing %q.\nOutput:\n%s", sub, out)
		}
	}
}

func TestRenderCodedDiagnostic(t *testing.T) {
	src := []byte("let x = 1\n")
	d := ErrorfCoded(mkSpan("a.esq", 1, 5, 1, 6), "E0001", "duplicate name %q", "x")
	out := d.Render(src)
	if !strings.Contains(out, "error[E0001]: duplicate name \"x\"") {
		t.Fatalf("missing coded header in:\n%s", out)
	}
}

func TestBag(t *testing.T) {
	var b Bag
	if b.HasErrors() {
		t.Fatal("empty bag should not have errors")
	}
	b.Add(Errorf(mkSpan("a.esq", 1, 1, 1, 2), "first"))
	b.Add(&Diagnostic{Severity: SevWarning, Primary: Label{Span: mkSpan("a.esq", 2, 1, 2, 2), Msg: "warn"}})
	b.Add(Errorf(mkSpan("a.esq", 3, 1, 3, 2), "second"))
	if !b.HasErrors() {
		t.Fatal("bag should have errors")
	}
	if got, want := b.First().Primary.Msg, "first"; got != want {
		t.Fatalf("First() = %q want %q", got, want)
	}
	if len(b.Diags) != 3 {
		t.Fatalf("len(Diags) = %d want 3", len(b.Diags))
	}
}

func TestSeverityString(t *testing.T) {
	cases := map[Severity]string{
		SevError:   "error",
		SevWarning: "warning",
		SevNote:    "note",
	}
	for s, want := range cases {
		if got := s.String(); got != want {
			t.Errorf("Severity(%d).String() = %q want %q", int(s), got, want)
		}
	}
}

func TestPosString(t *testing.T) {
	p := Pos{File: "a.esq", Line: 7, Col: 3}
	if got := p.String(); got != "a.esq:7:3" {
		t.Fatalf("got %q", got)
	}
}

func TestRenderSpanCrossLineClamps(t *testing.T) {
	// When the span runs past end-of-line, we should still render without panicking.
	src := []byte("xx\nyy\n")
	d := Errorf(mkSpan("a.esq", 1, 1, 2, 1), "spans two lines")
	out := d.Render(src)
	if !strings.Contains(out, "1 | xx") {
		t.Fatalf("expected first-line snippet, got:\n%s", out)
	}
}
