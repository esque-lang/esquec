---
title: Loop primitives
---

# Loop primitives

esque has no `for` keyword, no `while` keyword, no `break`, and no
`continue`. Every loop pattern that you would write in those keywords
in C has a named primitive in esque that says what shape the loop is.

The whole set:

| Primitive          | What it expresses                            |
|--------------------|----------------------------------------------|
| `lo..hi` / `lo..=hi` | iota — produce `[lo, lo+1, …]` as a tensor |
| `tabulate(N, f)`     | build a length-`N` tensor from `f(i)`      |
| `scan(init, f, v)`   | running prefix accumulator                  |
| `iterate_until(init, step, pred, max)` | bounded fixpoint            |
| `each(v, f)`         | side-effecting iteration (effect-checked)  |
| `+/(...)`            | reduction (special-cased, see prev page)   |

This page is a tour of all of them.

## Ranges: `lo..hi` and `lo..=hi`

```esque
0..5      // i32[5] = [0, 1, 2, 3, 4]
1..=5     // i32[5] = [1, 2, 3, 4, 5]
```

A range expression is a tensor literal expressed as bounds. Use it
wherever a tensor is expected:

```esque
+/(0..5)              // 10
+/(0..=5)             // 15
```

Both bounds must be integer literals today (or constants the compiler
can fold). Dynamic-bound ranges remain
[planned](/reference/planned/roadmap); until then you can fake it
with `tabulate(n, |i| i)` whose `n` is also a literal.

The size of a range is capped at 1 << 20 elements so a typo does not
generate a megabyte of `.rodata`.

## `tabulate(N, |i| ...)`

`tabulate` builds a length-`N` rank-1 tensor by calling `f(0)` …
`f(N-1)`:

```esque
tabulate(5, |i| i * i)                // [0, 1, 4, 9, 16]
tabulate(8, |i| if i < 4 { 0 } else { 1 })   // [0,0,0,0,1,1,1,1]
```

Where C would write `for (int i = 0; i < N; i++) v[i] = f(i);`,
esque writes `let v = tabulate(N, |i| f(i));`.

When the lambda is pure arithmetic of `i`, the compiler unrolls and
constant-folds the whole result into `.rodata`. There is no loop in
the emitted code.

`N` must be a literal. For `N ≤ 32` the compiler unrolls; for larger
`N` it emits a runtime loop (`OpTabulateLoop`, v0.11).

## `scan(init, |a, x| ..., v)`

`scan` is a prefix accumulator. Given an init, a combine function,
and a tensor:

```esque
scan(0, |a, x| a + x, [1, 2, 3, 4])    // [1, 3, 6, 10]
```

The output has the same length as the input. Element `i` is the
fold of the first `i+1` input elements with `init`.

A typical use is a running sum or max:

```esque
let v        = [3, 1, 4, 1, 5, 9, 2, 6];
let prefixes = scan(0, |a, x| a + x, v);   // [3,4,8,9,14,23,25,31]
let running_max = scan(0, |a, x| if x > a { x } else { a }, v);
```

For `v` of length ≤ 32 the compiler unrolls the scan; for longer
inputs it emits a runtime loop (`OpScanLoop`, v0.11).

## `iterate_until(init, step, pred, max)`

The bounded fixpoint:

```esque
iterate_until(0, |s| s + 1, |s| s == 7, 10)    // 7
```

Read it as: start at `init`. Apply `step` repeatedly. After each
application, check `pred`. As soon as `pred(state)` is true, freeze.
Stop after `max` iterations no matter what.

`max` is required and is a hard cap — there is no infinite loop.

```esque
// Newton's method for sqrt(2), six iterations max.
fn main() -> i32 = {
    let close = iterate_until(
        2.0,
        |x| (x + 2.0 / x) / 2.0,
        |x| x * x - 2.0 < 0.001 && x * x - 2.0 > -0.001,
        6
    );
    close as i32                  // 1
}
```

Today `iterate_until` is always unrolled, so `max ≤ 32` and the
state must be a scalar. A real `OpIterateUntilLoop` for arbitrary
`max` and tensor state is on the roadmap. (Unlike `tabulate`,
`scan`, and `iterate`, which gained large-N runtime loops in v0.11.)

The inherited fixed-iteration sibling, `iterate(n, init, f)`, is
also available — it just runs `f` exactly `n` times:

```esque
let result = iterate(5, 1.0, |x| x * 2.0);   // 32.0
```

## `each(v, f)`

`each` is the only intentionally side-effecting loop. It iterates
over a tensor and calls `f` on each element for its effect:

```esque
@io fn main() -> i32 = {
    each(0..5, print_i32);    // prints 0\n1\n2\n3\n4\n
    0
}
```

`f` must be a named top-level function (no closures or captured
names) whose [effect set](/reference/planned/effects) fits the
enclosing function's. A pure `f` is accepted in any caller; an
`@io` `f` requires the caller to be `@io` too. Element types
must match.

## Why no `for` and `while`?

Three reasons:

1. **They are imprecise.** `for` and `while` cover dozens of
   different shapes. A reader has to study the body to know whether
   the loop is building a tensor, accumulating a sum, doing a
   fixpoint, or something else. Named primitives say up front.
2. **They obscure parallelism.** `tabulate` is data-parallel by
   construction. `for` is not; the compiler has to prove it. The
   former reads exactly the same to a programmer and to the backend.
3. **They invite local mutation.** With `for` you reach for `mut`,
   counters, and accumulators. With `tabulate` and `scan` you do not.
   The functional shape of the program is preserved into the IR.

If you are coming from C, the table at the top of this page is the
phrase book.

## Next: [Pattern matching](pattern-matching)
