---
title: Iterate until
---

# Iterate until

```esque
# iter.esq
# iterate_until(init, step, pred, max)
# Counts 0,1,2,..., predicate fires at 7; result frozen at 7.
fn main() -> i32 = iterate_until(0, |s| s + 1, |s| s == 7, 10)
```

```bash
$ ./esquec build iter.esq -o iter
$ ./iter; echo $?
7
```

## What iterate_until does

Given:

- `init: T`
- `step: T -> T`
- `pred: T -> bool`
- `max: i32` literal

`iterate_until(init, step, pred, max)` repeatedly applies `step` to
the state. After each application, `pred` is checked; if true, the
state freezes. Either way, exactly `max` step applications happen
under the hood today (see below). The final state is the result.

The contract is:

> **The result is what `state` would be in a real loop that ran
> `step` until `pred(state)` was true, capped at `max` iterations.**

## Why a freeze instead of a `break`?

Because the compiler unrolls `iterate_until` at compile time. It
runs all `max` step applications statically and uses an `OpSelect`
cascade to preserve the first state value at which `pred` fired.
The cost is O(max) work in the emitted code — fine for `max ≤ 32`.
Unlike `tabulate`, `scan`, and `iterate` (which gained large-N
runtime loops in v0.11), `iterate_until` is still always unrolled,
so its `max` cap is real.

The general case (`OpIterateUntilLoop` with a real conditional jump)
is on the [roadmap](/reference/planned/roadmap). Once that lands,
the freeze and the early-break form will produce identical results;
the latter will just be cheaper.

## A more realistic example: Newton's method

```esque
# sqrt2.esq
fn main() -> i32 = {
    # Approximate sqrt(2) starting from x0 = 2.
    # Newton step: x' = (x + 2/x) / 2
    # Stop when x*x is within 0.01 of 2.0.
    let s = iterate_until(
        2.0,
        |x| (x + 2.0 / x) / 2.0,
        |x| {
            let e = x * x - 2.0;
            if e < 0.01 && e > -0.01 { true } else { false }
        },
        6
    );
    s as i32          # 1 (since sqrt(2) ≈ 1.414...)
}
```

The `iterate_until` form makes the *bound* explicit and obvious. A
reader can see at a glance that this can do at most six Newton
steps; there is no `while` to read carefully.

## Limits today

- `max` must be a literal, `≤ 32`.
- The state `T` must be a scalar (no tensor states yet).
- The step and predicate functions are inlined; they cannot recurse.

These are unrolling artefacts and lift with `OpIterateUntilLoop`.
