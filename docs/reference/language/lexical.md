---
title: Lexical structure
---

# Lexical structure

## Source encoding

esque source files are UTF-8. The lexer is byte-level for the ASCII
subset and validates multi-byte sequences only inside string and
char literals.

## Whitespace and comments

| Form                | Effect                          |
|---------------------|---------------------------------|
| ` `, `\t`, `\n`, `\r` | Whitespace; separates tokens. |
| `// ...`            | Line comment, to end of line.   |
| `/* ... */`         | Block comment, **nestable**.    |

## Identifiers

```
identifier = [a-zA-Z_][a-zA-Z0-9_]*
```

Identifiers are case-sensitive. There is no length limit other than
practicality.

## Keywords

The following are reserved and cannot be used as identifiers:

```
fn   return   let   if   else   match   true   false   in   as   mut
```

`mut` is reserved for the future linear-types/mutability work; it is
not currently meaningful. `return` is recognised as a keyword but
is rarely needed: every function body is a single expression, so
`return e` is equivalent to writing `e` as the trailing expression
of the body block.

The C-style loop keywords `for`, `while`, `break`, and `continue`
are **not** keywords in esque. They lex as ordinary identifiers and
can be used as variable names. (See
[Loop primitives](loop-primitives) for the alternative.)

## Integer literals

```
42
1_000_000        // underscores allowed
99_i32           // explicit type suffix
'A'              // char literal: i32 codepoint
```

| Suffix | Type | Codegen status |
|--------|------|----------------|
| (none) | `i32` (default) | supported |
| `_i32` | `i32`           | supported |
| `_i64`, `_u32` | as named | supported (v0.11; scalar) |
| `_i8`, `_u8`   | as named | supported (v0.12; scalar) |
| `_i16`, `_u16`, `_u64` | as named | type-checks; backend rejects |

A char literal `'X'` is an integer literal whose value is the
Unicode codepoint of the character. Standard escapes (`\n`, `\t`,
`\\`, `\'`, `\"`, `\0`, `\xNN`, `\u{XXXX}`) are recognised.

## Float literals

```
3.14
0.0
1e-5
3.14_f64
```

A float literal must include a decimal point or an exponent.
`3.` is not a float literal; write `3.0`.

| Suffix | Type | Codegen status |
|--------|------|----------------|
| (none) | `f32` (default) | supported |
| `_f32` | `f32`           | supported |
| `_f64` | `f64`           | supported (v0.12; scalar) |

## Boolean literals

```
true   false
```

## String literals

`"..."` lexes as a `TkString` token and is admitted by the type
checker as a value of type `string` — an immutable UTF-8 fat
pointer of `(ptr, len)`. As of v0.12, string literals can be
threaded through `let`s and passed to `print_str`. The full string
surface (concatenation, indexing, runtime construction) is still
[planned](/reference/planned/strings).

## Operators (token list)

```
+  -  *  /  %                       (scalar arithmetic)
.+ .- .* ./ .%                      (element-wise tensor arithmetic)
@                                   (matrix multiply, parsed)
+/  */  -/  //                      (reductions; lexed as op + '/')
==  !=  <  <=  >  >=                (comparisons)
&&  ||  !                           (logical)
=                                   (let / fn body introducer)
->  =>                              (function ret, match arm)
,  ;  :  .                          (separators)
( ) [ ] { }                         (groupings)
|>                                  (pipeline)
|...|                               (lambda parameter brackets)
..  ..=                             (range)
```

See [Operators and precedence](operators) for the precedence table.
