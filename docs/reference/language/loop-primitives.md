---
title: Loop primitives
---

# Loop primitives

esque has no `for`, no `while`, no `break`, no `continue`. This page
is the formal reference for the loop primitives that replace them.

## Range

```
lo..hi      // i32[hi - lo]   exclusive
lo..=hi     // i32[hi - lo + 1] inclusive
```

**Constraints:**

- Both bounds must be `*ast.IntLit`. Dynamic bounds are rejected
  with a diagnostic; const-eval for non-literal bounds is planned.
- `hi - lo > 0`. Empty / reversed ranges error.
- Total size ≤ 1 << 20 elements.

**Lowering:** desugars to an `i32` `OpTensorLit` of `OpConstInt`s.
Constant tensors hit the `.rodata` fast path.

## tabulate

```
tabulate(N, f) -> T[N]
```

| Position | Type / form                       |
|----------|-----------------------------------|
| `N`      | `*ast.IntLit`, `0 < N ≤ 1 << 20`  |
| `f`      | 1-arg lambda or named fn `i32 -> T` |

**Lowering:** for `N ≤ 32` the lambda body is unrolled `f(0)` …
`f(N-1)` and collected into an `OpTensorLit`. If every result is a
constant, the literal hits `.rodata`. For `N > 32` CEIR emits an
`OpTabulateLoop` and MIR lowers it to a counted runtime loop
(v0.11).

## scan

```
scan(init: T, f: (T, T) -> T, v: T[N]) -> T[N]
```

The combine function takes the running accumulator first. The
output's element `i` is
`f(... f(f(init, v[0]), v[1]) ..., v[i])`.

**Constraints:** `N ≤ 1 << 20`. For `N ≤ 32` the scan unrolls; for
`N > 32` CEIR emits an `OpScanLoop` lowered to a runtime loop. Init
must have type `T` matching the element type of `v` and the
input/output types of `f`.

## iterate_until

```
iterate_until(init: T, step: T -> T, pred: T -> bool, max: i32) -> T
```

| Position | Constraint                                  |
|----------|---------------------------------------------|
| `init`   | scalar `T` (no tensor states yet)           |
| `step`   | 1-arg fn/lambda `T -> T`                    |
| `pred`   | 1-arg fn/lambda `T -> bool`                 |
| `max`    | `*ast.IntLit`, `0 < max ≤ 32` (unroll only) |

**Semantics:** the result is what `state` would be in a real loop
that ran `step` until `pred(state)` was true, capped at `max`
iterations.

**Lowering:** unrolled select-cascade. Iteration `i` computes
`s_i_step = step(s_{i-1})`, `p_i = pred(s_i_step)`, then
`s_i = select(p_done, s_{i-1}, s_i_step)`. After `max` iterations
the freeze gives the equivalent of breaking on first true. Cost is
O(max) regardless of when `pred` first fires.

The general `OpIterateUntilLoop` (real branch, arbitrary `max`,
tensor state) is a roadmap item.

## iterate

```
iterate(n: i32, init: T, f: T -> T) -> T
```

Run `f` exactly `n` times starting from `init`. Returns the final
state. Equivalent to `iterate_until(init, f, |_| false, n)`, just
without the predicate machinery.

**Constraints:** `n` must be a literal, `n ≤ 1 << 20`. For `n ≤ 32`
the body is unrolled; for `n > 32` CEIR emits an `OpIterateLoop`
lowered to a runtime loop.

## each

```
each(v: T[N], f: T -> unit) -> unit
```

Iterate side-effectingly. As of v0.13 `f` must be a named function
whose [effect set](/reference/planned/effects) is a subset of the
enclosing function's effects:

- A pure `f` (no annotation) is accepted in any caller.
- An `@io` `f` (e.g. `print_i32`, `print_f32`, `print_str`, or any
  user-defined `@io` wrapper) requires the enclosing function to be
  `@io` as well.

A closure or captured name is still a type error — `f` must resolve
to a top-level function name. The pre-v0.13 hardcoded
`{print_i32, print_f32}` allowlist is gone.

## Scope and limits today

`tabulate`, `scan`, `iterate`, and `each` accept counts up to
`1 << 20`. At or below 32 the body is unrolled inline; above 32
CEIR emits the corresponding `OpTabulateLoop` / `OpScanLoop` /
`OpIterateLoop` / `OpEachLoop` and MIR lowers it to a counted
runtime loop (v0.11).

`iterate_until` is the exception: because its termination depends
on a runtime predicate, the unroll-and-select-cascade lowering is
the only form today, and `max` is capped at 32. The general
`OpIterateUntilLoop` (real branch on the predicate) is on the
roadmap.
