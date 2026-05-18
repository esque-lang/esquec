---
title: Lexer
---

# Lexer

`internal/lex`. A hand-written byte-level UTF-8 lexer.

## Token shape

```go
type Token struct {
    Kind Kind        // TkInt, TkIdent, TkPlus, ...
    Lit  string      // raw lexeme
    Span diag.Span   // source location
}
```

`Kind` is an enum (`internal/lex/token.go`). `Lit` is the raw
lexeme — for an `IntLit` like `1_000_i32` it is the literal text
including underscores and suffix; semantic interpretation happens
in the parser.

## Categories

| Category    | Examples |
|-------------|----------|
| Keywords    | `fn`, `return`, `let`, `if`, `else`, `match`, `true`, `false`, `in`, `as`, `mut` |
| Identifiers | `foo`, `_bar`, `MyType`. (`for`, `while`, `break`, `continue` lex as identifiers — not keywords.) |
| Literals    | `42`, `1_000`, `42_i32`, `'A'`, `3.14`, `"..."`, `true` |
| Operators   | `+`, `.+`, `\|>`, `==`, `..`, `..=`, `=>`, `->` |
| Punctuation | `(`, `)`, `[`, `]`, `{`, `}`, `;`, `,`, `:`, `\|` |

## Notable design choices

- **Byte-level for ASCII, validated for non-ASCII inside string and
  char literals.** This avoids runtime regex but accepts the full
  Unicode codepoint range inside `'...'`.
- **Block comments are nestable**: `/* /* nested */ still inside */`.
  This is a deliberate divergence from C.
- **Spans are byte offsets** into the original source, plus a 1-based
  line/column for diagnostics. The diagnostic renderer maps them
  back to the source line.

## Why `for`, `while`, `break`, `continue` are not keywords

esque does not have C-style control flow; the loop primitives
(`tabulate`, `scan`, `iterate`, `iterate_until`, `each`) replace
them. To keep the surface small and let those identifiers serve as
ordinary variable names, the lexer treats `for`, `while`, `break`,
and `continue` as plain identifiers.
