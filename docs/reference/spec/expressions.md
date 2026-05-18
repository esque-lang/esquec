---
title: Expressions
---

# Spec: expressions

esque is expression-oriented. Almost everything has a value.

## Literals

Integer, float, bool, char, tensor, and string literals are values.
String literals have type `string` — see
[Lexical → String literals](lexical) for the surface that has
shipped and [Planned: strings](/reference/planned/strings) for the
rest.

## Bindings

`let` introduces an immutable binding. There is no `mut` form
today; that is part of the
[planned linear types](/reference/planned/linear-types).

Two forms exist: a statement form usable inside a block, and an
expression form usable anywhere a value is expected.

### Statement form

```
{
    let x = e;       // immutable
    let x: T = e;    // immutable, with explicit type
    body
}
```

### Expression form

```
let x = e in body
let x: T = e in body
```

Right-associative. The body sees `x` in scope. Multiple bindings
chain by nesting:

```
fn k() -> i32 =
    let a = 10 in
    let b = a * 2 in
    a + b
```

The expression form is sugar for `{ let x = e; body }`, so scoping
and shadowing follow the block rules above.

## Arithmetic and element-wise operators

Scalar operators work on rank-0 numeric values:

```
+  -  *  /  %
```

Element-wise operators work on tensors of equal type and shape:

```
.+  .-  .*  ./  .%
```

Broadcasting between mismatched shapes is **(planned)**; today the
two operands of an element-wise op must have equal shapes.

The `@` operator is reserved for matrix multiplication; it parses
but does not yet codegen.

## Reductions

```
<op>/(v)
```

`<op>/` reduces a tensor to a scalar by left-folding `op`:

```
+/(v)    // sum
*/(v)    // product
-/(v)    // running difference (rare; usually use scan)
```

The reduction operator is parametric in the operator: any binary
operator the element type defines can be used. Axis-aware reduction
(`+/[axis=0]`) is **(planned)**.

## Comparisons and logical operators

```
== != < <= > >=     // any equality-supporting T, T -> bool
&& ||               // bool, bool -> bool
!                   // bool -> bool
```

## Pipeline

```
x |> f          // = f(x)
x |> f(y, z)    // = f(x, y, z)
```

Left-associative; precedence 1 (the loosest binary op). The piped
value always becomes the first argument.

## Range expressions

```
lo..hi      // exclusive
lo..=hi     // inclusive
```

A range expression evaluates to a rank-1 `i32` tensor of the
expected length. Both bounds must be integer literals today;
dynamic bounds (with const-eval) are **(planned)**.

## Tabulate

```
tabulate(N, |i| f(i))
```

Evaluates `f(0)`, `f(1)`, …, `f(N-1)` and packs the results into a
rank-1 tensor of length `N`. `N` must be a literal in
`1..=1<<20`. For `N ≤ 32` the body is unrolled; for `N > 32` the
implementation emits a counted runtime loop (`OpTabulateLoop`,
shipped in v0.11).

## Scan

```
scan(init, |a, x| f(a, x), v)
```

Returns a tensor of the same shape as `v` whose element `i` is
`f(... f(f(init, v[0]), v[1]) ..., v[i])`. `N ≤ 1<<20`; for
`N > 32` the implementation emits `OpScanLoop` (shipped in v0.11).

## Iterate-until

```
iterate_until(init, |s| step(s), |s| pred(s), max)
```

Returns the value `state` would have in a real loop that ran
`step` until `pred(state)` was true, capped at `max` iterations.
`max` must be a literal `≤ 32` and the state must be scalar today;
unlike the other primitives, the runtime-loop counterpart
(`OpIterateUntilLoop`, with predicate-driven branching) is still
**(planned)**.

## Iterate

```
iterate(n, init, |s| step(s))
```

Run `step` exactly `n` times starting from `init`. Equivalent to
`iterate_until(init, step, |_| false, n)` but cheaper because there
is no predicate to thread. `n` must be a literal in `1..=1<<20`;
for `n > 32` the implementation emits `OpIterateLoop` (shipped in
v0.11).

## Each

```
each(v, f)
```

Iterate side-effectingly. As of v0.13, `f` must be a named function
whose [effect set](/reference/planned/effects) fits the enclosing
function's. Pure `f` is accepted everywhere; `@io` `f` requires an
`@io` caller. Result type: `unit`.

## Conditionals

```
if cond { e1 } else { e2 }
if cond { e1 } else if cond' { e2 } else { e3 }
```

The condition has type `bool`. Both arms must have a common type,
which is the type of the whole `if`. The `else` is required when
the `if` is used for its value.

## Match

```
match e {
    pattern1 => e1,
    pattern2 if guard => e2,
    _ => e3,
}
```

Arms are tried top-to-bottom; the first match wins. Patterns
supported today:

- Integer literals (`0`, `-1`, `'A'`).
- Boolean literals (`true`, `false`).
- Identifier (binds the scrutinee).
- `_` (wildcard).

Exhaustiveness checking is **(planned)**.

## Blocks

```
{ stmt; stmt; ...; expr }
```

A block evaluates each statement in order, then evaluates the final
expression and produces its value. A block whose final form is a
statement (`expr;`) has type `unit`.

## Removed forms

Earlier drafts of the spec included:

- `for i in 0..N { body }` — replaced by `tabulate(N, |i| body)`,
  by `each(0..N, f)`, or by `+/(tabulate(N, |i| body))` for
  reductions.
- `while cond { body }` — replaced by
  `iterate_until(init, step, |s| !cond(s), max)`.
- Comprehensions `[[ f(i, j) for j in 0..N ] for i in 0..M ]` —
  replaced by `tabulate(M, |i| tabulate(N, |j| f(i, j)))`.
- `mut` and `+=` family — to be reintroduced under linear types.
