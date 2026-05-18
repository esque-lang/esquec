// Package diag provides source spans and diagnostic reporting.
package diag

import (
	"fmt"
	"strings"
)

// Pos is a 1-indexed source position.
type Pos struct {
	File    string
	Line    int
	Col     int
	ByteOff int
}

func (p Pos) String() string {
	return fmt.Sprintf("%s:%d:%d", p.File, p.Line, p.Col)
}

// Span covers a half-open range [Start, End) in a source file.
type Span struct {
	Start Pos
	End   Pos
}

func (s Span) String() string { return s.Start.String() }

// Severity is the severity of a diagnostic.
type Severity int

const (
	SevError Severity = iota
	SevWarning
	SevNote
)

func (s Severity) String() string {
	switch s {
	case SevError:
		return "error"
	case SevWarning:
		return "warning"
	case SevNote:
		return "note"
	}
	return "unknown"
}

// Label is a span with a message attached. Used as the primary site of a
// diagnostic and as additional notes.
type Label struct {
	Span Span
	Msg  string
}

// Diagnostic is a structured compiler diagnostic.
type Diagnostic struct {
	Severity Severity
	Code     string // e.g. "E0123"; empty if uncoded
	Primary  Label
	Notes    []Label
	Help     string
}

// Error implements the error interface, returning a single-line summary so
// the diagnostic can flow through Go's error machinery without being
// reformatted.
func (d *Diagnostic) Error() string {
	hdr := fmt.Sprintf("%s: %s: %s", d.Primary.Span, d.Severity, d.Primary.Msg)
	if d.Code != "" {
		hdr = fmt.Sprintf("%s: %s[%s]: %s", d.Primary.Span, d.Severity, d.Code, d.Primary.Msg)
	}
	return hdr
}

// Render produces a human-friendly multi-line rendering with caret-style
// snippets. `src` is the contents of the file referenced in the primary
// span; if nil, snippets are omitted.
func (d *Diagnostic) Render(src []byte) string {
	var b strings.Builder
	if d.Code != "" {
		fmt.Fprintf(&b, "%s[%s]: %s\n", d.Severity, d.Code, d.Primary.Msg)
	} else {
		fmt.Fprintf(&b, "%s: %s\n", d.Severity, d.Primary.Msg)
	}
	fmt.Fprintf(&b, "  --> %s\n", d.Primary.Span.Start)
	if src != nil {
		writeSnippet(&b, src, d.Primary.Span, "")
	}
	for _, n := range d.Notes {
		fmt.Fprintf(&b, "note: %s\n", n.Msg)
		fmt.Fprintf(&b, "  --> %s\n", n.Span.Start)
		if src != nil {
			writeSnippet(&b, src, n.Span, "")
		}
	}
	if d.Help != "" {
		fmt.Fprintf(&b, "help: %s\n", d.Help)
	}
	return b.String()
}

// writeSnippet writes a single-line caret-style snippet for the given span.
// If the span crosses lines, only the first line is shown with carets to the
// end-of-line.
func writeSnippet(b *strings.Builder, src []byte, sp Span, label string) {
	if sp.Start.Line < 1 {
		return
	}
	// Find the start of the line containing sp.Start.
	line := sp.Start.Line
	off := lineStart(src, line)
	if off < 0 {
		return
	}
	end := off
	for end < len(src) && src[end] != '\n' {
		end++
	}
	lineText := string(src[off:end])
	prefix := fmt.Sprintf("%4d | ", line)
	fmt.Fprintf(b, "%s%s\n", prefix, lineText)
	// Caret line.
	caretCol := sp.Start.Col - 1
	if caretCol < 0 {
		caretCol = 0
	}
	caretLen := 1
	if sp.End.Line == sp.Start.Line && sp.End.Col > sp.Start.Col {
		caretLen = sp.End.Col - sp.Start.Col
	}
	if caretCol+caretLen > len(lineText) {
		caretLen = len(lineText) - caretCol
		if caretLen < 1 {
			caretLen = 1
		}
	}
	fmt.Fprintf(b, "     | %s%s", strings.Repeat(" ", caretCol), strings.Repeat("^", caretLen))
	if label != "" {
		fmt.Fprintf(b, " %s", label)
	}
	b.WriteString("\n")
}

// lineStart returns the byte offset of the start of the given 1-indexed
// line in src, or -1 if the line is out of range.
func lineStart(src []byte, line int) int {
	if line == 1 {
		return 0
	}
	cur := 1
	for i := 0; i < len(src); i++ {
		if src[i] == '\n' {
			cur++
			if cur == line {
				return i + 1
			}
		}
	}
	return -1
}

// Error is retained for backwards compatibility. It is a Diagnostic-shaped
// alias used while migrating older call sites.
type Error = Diagnostic

// Errorf constructs an error diagnostic.
func Errorf(s Span, format string, args ...any) *Diagnostic {
	return &Diagnostic{
		Severity: SevError,
		Primary:  Label{Span: s, Msg: fmt.Sprintf(format, args...)},
	}
}

// ErrorfCoded constructs a coded error diagnostic.
func ErrorfCoded(s Span, code, format string, args ...any) *Diagnostic {
	return &Diagnostic{
		Severity: SevError,
		Code:     code,
		Primary:  Label{Span: s, Msg: fmt.Sprintf(format, args...)},
	}
}

// WithNote attaches a note to a diagnostic. Returns the diagnostic for chaining.
func (d *Diagnostic) WithNote(s Span, format string, args ...any) *Diagnostic {
	d.Notes = append(d.Notes, Label{Span: s, Msg: fmt.Sprintf(format, args...)})
	return d
}

// WithHelp attaches a help message. Returns the diagnostic for chaining.
func (d *Diagnostic) WithHelp(format string, args ...any) *Diagnostic {
	d.Help = fmt.Sprintf(format, args...)
	return d
}

// Bag accumulates diagnostics during a compile pass. It supports parser
// recovery (multiple diagnostics) without losing the first error semantics
// of the existing code.
type Bag struct {
	Diags []*Diagnostic
}

// Add appends a diagnostic.
func (b *Bag) Add(d *Diagnostic) { b.Diags = append(b.Diags, d) }

// HasErrors reports whether any diagnostic has SevError.
func (b *Bag) HasErrors() bool {
	for _, d := range b.Diags {
		if d.Severity == SevError {
			return true
		}
	}
	return false
}

// First returns the first error diagnostic, or nil.
func (b *Bag) First() *Diagnostic {
	for _, d := range b.Diags {
		if d.Severity == SevError {
			return d
		}
	}
	return nil
}

// Render renders all diagnostics as a single string.
func (b *Bag) Render(src []byte) string {
	var sb strings.Builder
	for _, d := range b.Diags {
		sb.WriteString(d.Render(src))
		sb.WriteString("\n")
	}
	return sb.String()
}
