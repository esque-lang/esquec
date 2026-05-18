---
title: Pattern matching
---

# Pattern matching

```
match scrutinee {
    pattern1 => expr1,
    pattern2 if guard => expr2,
    _ => default
}
```

A `match` is an expression. Arms are tried top-to-bottom; the first
match wins. The scrutinee is evaluated once. All arm right-hand
sides must have the same type.

## Patterns

| Pattern        | Matches                                  |
|----------------|------------------------------------------|
| integer literal | the value (e.g. `0`, `1`, `-42`)        |
| char literal    | the codepoint as `i32` (e.g. `'A'`)     |
| `true` / `false`| the corresponding bool                  |
| identifier `n`  | binds the scrutinee to a fresh name `n` |
| `_`             | wildcard, no binding                    |

There are no constructor patterns, no struct destructuring, no
tensor-shape patterns. These arrive with ADTs / records.

## Guards

An arm may carry an `if guard` after its pattern:

```
match x {
    n if n > 0 => 1,
    _ => -1
}
```

The arm fires only when both pattern matches and the guard
evaluates to `true`. The bound name (`n` here) is in scope inside
the guard and the right-hand side.

## Exhaustiveness

The type checker does **not** verify exhaustiveness today. A
`match` whose patterns do not collectively cover every possible
input compiles fine, but feeding it an unmatched value at runtime
is undefined.

Always end a `match` with a wildcard arm until exhaustiveness
checking lands. See
[planned: roadmap](/reference/planned/roadmap).

## Typing

- The scrutinee has some type `T`.
- Each pattern must be compatible with `T`: integer patterns
  require `i32`, bool patterns require `bool`, wildcard / binding
  patterns match any `T`.
- Each guard must have type `bool`.
- All right-hand sides have a common type, which is the type of the
  whole match expression.
