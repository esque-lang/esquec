---
title: Pattern matching
---

# Pattern matching

`match` chooses between branches by matching a scrutinee against
patterns. It is the structured form of long `if`/`else if` chains.

```esque
fn classify(n: i32) -> i32 = match n {
    0 => 10,
    1 => 20,
    2 => 30,
    _ => 99
}
```

Every `match` has the same shape:

- One scrutinee expression (`n` above).
- A list of arms `pattern => expression,`.
- Arms are tried top-to-bottom.
- The first match wins.
- A wildcard `_` matches anything.

## Patterns supported today

| Pattern        | Matches                                  |
|----------------|------------------------------------------|
| `42`           | the integer literal `42`                 |
| `'A'`          | the i32 codepoint `65`                   |
| `true`/`false` | the corresponding bool                   |
| `n`            | binds the scrutinee to a fresh name `n`  |
| `_`            | wildcard, no binding                     |

## Guards

An arm can have a guard with `if`:

```esque
fn sign(x: i32) -> i32 = match x {
    0 => 0,
    n if n > 0 => 1,
    _ => -1
}
```

The arm only fires when both the pattern matches and the guard is
true. The bound name (`n`) is in scope inside both the guard and the
right-hand side.

## Exhaustiveness

esque does not yet check exhaustiveness. If you forget the wildcard
arm and feed in a value none of the patterns matches, you get a
runtime error at the missing-arm point. Exhaustiveness checking is
[planned](/reference/planned/roadmap) and pairs with the future ADT
support; for now, end every `match` with `_ => ...`.

## When to use `match` vs `if`

- Use `if`/`else if` when there are 1–2 conditions and they are
  independent.
- Use `match` when you are dispatching on a value (especially a
  small integer constant set, an enum-shaped int, or a guard chain
  on the same variable).

The two compile to similar code; pick whichever reads better.

## Tensor patterns?

Not yet. There is no destructuring on tensor shapes. When ADTs and
records arrive (see the [roadmap](/reference/planned/roadmap)) so
will their patterns.

## Next: [Control flow](control-flow)
