---
title: Diagnostics
---

# Diagnostics

`internal/diag`. Structured, position-aware error reporting.

## Types

```go
type Pos struct {
    File    string
    Line    int       // 1-based
    Col     int       // 1-based
    ByteOff int
}

type Span struct {
    Start, End Pos
}

type Label struct {
    Span Span
    Msg  string
}

type Severity int  // SevError | SevWarning | SevNote

type Diagnostic struct {
    Severity Severity
    Code     string  // e.g. "E0123"; empty if uncoded
    Primary  Label   // primary location and message
    Notes    []Label // optional secondary spans + messages
    Help     string  // optional follow-up suggestion
}
```

A `Diagnostic` is an `error`; its `Error()` method returns a
single-line summary built from `Severity`, optional `Code`, and the
primary span and message. The driver's `reportErr` uses `errors.As`
to detect it and call `Render(src)` for the caret-style display;
non-diagnostic errors fall back to plain `%v`.

## Caret rendering

```
test.esq:3:5
  3 │   fn main() -> i32 = 5000000000
    │                      ^^^^^^^^^^ integer literal overflows i32
    │                      help: try _i64
```

`Render(src)` is responsible for:

- Reading the source line.
- Drawing carets under `Span`.
- Indenting the help and note blocks.

## When to produce a `Diagnostic`

Anywhere the user would benefit from a span-anchored error:

- Parse errors (already structured).
- Type-check errors (already structured).
- CEIR/MIR/x86 lowering errors that surface to the user (e.g.
  "tabulate(N) requires N ≤ 32").

When in doubt, emit a `*diag.Diagnostic` rather than
`fmt.Errorf`. The driver will render it for free.

## Style

- **Primary message**: a sentence fragment, no trailing period,
  present tense. e.g. "integer literal overflows i32".
- **Help**: a forward-pointing suggestion, often referencing a
  planned feature. e.g. "try `_i64`".
- **Notes**: optional secondary `Label`s for things like "first
  defined here" / "first used here". Each note carries its own
  span and message.
