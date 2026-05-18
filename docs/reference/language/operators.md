---
title: Operators and precedence
---

# Operators and precedence

esque uses a Pratt parser with explicit precedence numbers. The
following table is the canonical list, sorted from loosest to
tightest binding. **Higher number = tighter binding.**

## Infix table

| Prec | Operators                          | Associativity   | Group           |
|------|------------------------------------|-----------------|-----------------|
| 1    | `\|>`                              | left            | pipeline        |
| 5    | `\|\|`                             | left            | logical or      |
| 6    | `&&`                               | left            | logical and     |
| 10   | `==` `!=` `<` `<=` `>` `>=`        | non-associative | comparison      |
| 15   | `..` `..=`                         | non-associative | range           |
| 20   | `+` `-` `.+` `.-`                  | left            | additive        |
| 30   | `*` `/` `%` `.*` `./` `.%`         | left            | multiplicative  |
| 35   | `@`                                | left            | matrix multiply |
| 40   | (function call, indexing, casts)   | left            | postfix         |

`|>` is the loosest infix operator — it binds looser than every
arithmetic, logical, and comparison operator, so
`x + 1 |> f` parses as `(x + 1) |> f` and `x |> f && y` parses as
`(x |> f) && y`.

## Prefix operators

| Operator | Effect                                |
|----------|---------------------------------------|
| `-`      | Numeric negation                      |
| `!`      | Logical not                           |
| `+/`     | Reduction by `+` (sum)                |
| `*/`     | Reduction by `*` (product)            |
| `-/`     | Reduction by `-`                      |

`<op>/` is read as the prefix reduction operator; the `op` part is
any binary operator the element type defines.

## Postfix forms

- **Function call**: `f(args)`
- **Cast**: `e as T`

There is no postfix `?`, no postfix `!`, no field access (yet — no
record types).

## Examples by precedence

```esque
a + b * c          // a + (b * c)
a .* b .+ c        // (a .* b) .+ c
0..n + 1           // 0..(n + 1)         (.. is precedence 15, + is 20)
a < b && c < d     // (a < b) && (c < d)
x |> f |> g        // g(f(x))             (left-assoc)
x as i32 + 1       // (x as i32) + 1
```

## Notes on element-wise vs scalar

`.+`, `.-`, `.*`, `./`, `.%` exist only as tensor element-wise
operators. They cannot be used on scalars.

`+`, `-`, `*`, `/`, `%` exist only on scalars (and shape arithmetic
inside type-position expressions).

The `@` matrix-multiply operator is parsed but not yet codegened.
