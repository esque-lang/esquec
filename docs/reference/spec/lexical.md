---
title: Lexical
---

# Spec: lexical structure

## Source encoding

esque source files are UTF-8. Implementations must accept any
well-formed UTF-8 byte sequence; they may reject ill-formed
sequences or, inside string and char literals, render them as a
diagnostic.

## Whitespace

The whitespace characters are space (`U+0020`), tab (`U+0009`),
newline (`U+000A`), and carriage return (`U+000D`). Whitespace is
significant only as a token separator.

## Comments

```
# line comment, ends at newline
/* block comment, nestable */
```

Comments are equivalent to whitespace. The line-comment lead-in is
`#` as of v0.14; `//` is reserved for the divide-reduction operator
(see [Expressions](expressions)).

## Tokens

| Class       | Description                          |
|-------------|--------------------------------------|
| Identifier  | `[a-zA-Z_][a-zA-Z0-9_]*`             |
| Keyword     | reserved identifier                  |
| IntLit      | integer literal, optional suffix     |
| FloatLit    | float literal, optional suffix       |
| BoolLit     | `true` or `false`                    |
| CharLit     | `'`, codepoint, `'`                  |
| StringLit   | `"`, characters, `"` (impl, literals only) |
| Operator    | one of the operator tokens           |
| Punctuation | `(`, `)`, `[`, `]`, `{`, `}`, `;`, `,`, `:`, `\|` |

### Reserved keywords

```
fn   return   let   if   else   match   true   false   in   as   mut
```

`mut` is reserved for the planned linear-types and mutability work;
it currently has no semantic effect. `return` is reserved but
rarely needed: a function body is a single expression, so the
trailing expression of the body is the return value.

The following are *not* keywords and parse as identifiers: `for`,
`while`, `break`, `continue`. The corresponding control-flow
patterns are expressed via the loop primitives (see
[Expressions](expressions)).

### Integer literals

```
IntLit       = digits ("_" digits)* IntSuffix?
             | "0x" hex ("_" hex)* IntSuffix?
IntSuffix    = "_i8" | "_i16" | "_i32" | "_i64"
             | "_u8" | "_u16" | "_u32" | "_u64"
```

Without a suffix, an integer literal is the type the context
expects (default `i32`). With a suffix, it is the named type.

### Float literals

```
FloatLit     = digits ("_" digits)* "." digits ("_" digits)* Exponent? FloatSuffix?
             | digits ("_" digits)* Exponent FloatSuffix?
Exponent     = ("e" | "E") ("+" | "-")? digits
FloatSuffix  = "_f32" | "_f64"
```

A float literal must contain either a decimal point or an exponent.

### Char literals

A `CharLit` is `'` followed by a single character or escape sequence
followed by `'`. Its value is an `i32` containing the Unicode
codepoint.

Recognised escapes: `\n`, `\t`, `\r`, `\\`, `\'`, `\"`, `\0`,
`\xNN` (8-bit), `\u{XXXX}` (Unicode codepoint).

### String literals

A `StringLit` is `"` followed by characters or escape sequences
followed by `"`. Its value has type `string` — an immutable UTF-8
fat pointer of `(ptr, len)`. As of v0.12 string literals are
admitted by the type checker and may be threaded through `let`s
and passed to `print_str`. The full string surface (concatenation,
indexing, runtime construction) is
[planned](/reference/planned/strings).

The same escape sequences as `CharLit` are recognised.
