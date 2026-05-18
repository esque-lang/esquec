---
title: Control flow
---

# Control flow

esque is expression-oriented. Every form of "control flow" is an
expression that evaluates to a value of a known type — there are no
statements, no `return`, and no fall-through.

## `if`/`else`

```esque
if x > 0 { 1 } else if x < 0 { -1 } else { 0 }
```

The two branches must have the same type. The `else` is **required**
unless the `if` is used purely for its side effect inside a block —
and esque has very few of those today, so in practice `else` is
always there.

```esque
fn abs(x: i32) -> i32 = if x < 0 { -x } else { x }
```

## Blocks `{ ... }`

A block is itself an expression. It evaluates each
semicolon-separated step in order, then evaluates the final
expression and produces its value:

```esque
fn main() -> i32 = {
    let a = 10;
    let b = 3;
    a + b              # last expression: this is the value of the block
}
```

The final expression has no trailing semicolon. If you write a
trailing semicolon you have made the block produce `unit`, which is
almost certainly not what you wanted.

## Sequencing side effects

A block is also where you put a side-effect call you want to discard
the result of:

```esque
fn main() -> i32 = {
    each(0..5, print_i32);    # side effect, returns unit
    0                          # exit code
}
```

`each(...)` returns `unit`; the semicolon after it discards the
value, and the final `0` is the value of the block.

## Loops

There are no loop *statements*. The
[loop primitives](loop-primitives) — `tabulate`, `scan`,
`iterate_until`, `each` — are all expressions. You bind their result
with `let` if you need it, or sequence them in a block if you only
care about side effects.

## Early exit

There is no `return` and no `break`. If you find yourself wanting
one, ask whether the function is doing two things at once and could
be split, or whether `iterate_until`'s built-in stop predicate is
what you actually wanted.

## Pattern-matching as control flow

`match` (covered on the [previous page](pattern-matching)) is the
primary structured-dispatch construct. Long `if`/`else if` chains on
the same scrutinee are usually clearer as a `match`.

## Next: [Printing and I/O](printing-and-io)
