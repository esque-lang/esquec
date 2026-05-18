---
title: Pipelines and reductions
---

# Pipelines and reductions

Two ideas in esque do most of the work of "shaping" a computation:
the **pipeline** operator `|>` and the **reduction** operator
`<op>/`. Both are pure desugarings, but their effect on how programs
read is large enough that they get their own page.

## The pipeline operator

```esque
x |> f         // = f(x)
x |> f(y)      // = f(x, y)
```

That is the whole rule. `|>` is left-associative so chains read top
to bottom:

```esque
3 |> double |> add_one |> square
// = square(add_one(double(3)))
```

When the pipelined expression already takes one argument, the
piped value is inserted *first*:

```esque
fn add(a: i32, b: i32) -> i32 = a + b
10 |> add(5)         // = add(10, 5) = 15
```

`|>` exists because most numeric pipelines look like a stack of
transformations on one value. Writing them inside-out (function-call
order) reads worse than top-to-bottom.

### When to reach for `|>`

- You are chaining at least three transformations.
- Each step is a meaningful name (`normalise`, `clamp`, `softmax`).
- The intermediate values do not need to be re-used.

If the next step needs to combine the piped value with two other
things you computed earlier, give it a name with `let` instead.

## The reduction operator

`<op>/` (read "op-over") reduces a tensor to a scalar by folding the
operator across the elements:

```esque
+/([1.0, 2.0, 3.0])     // 6.0
*/([1, 2, 3, 4])        // 24
```

Any binary operator that is defined on the element type works:

```esque
+/  -/  */  /
```

Of these, `+/` is the workhorse. It pairs with `.*` to give the dot
product:

```esque
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)
```

### Why a special operator and not just `reduce(+, v)`?

Because reductions are *the* hot path on tensor data. Giving them
syntax means the compiler can:

- emit AVX2 horizontal reductions (`vhaddps`) when the shape allows
- emit SSE3 `haddps` chains when AVX2 is not available
- fall back to a scalar accumulator for the tail of an irregular
  shape

You write `+/`. The backend picks the lanes.

### Reductions over `tabulate` and ranges

`+/` over a `tabulate` is the bread-and-butter shape of a numeric
program:

```esque
+/(tabulate(N, |i| f(i)))
```

That is "sum f(i) over i=0..N-1". When `f` is pure arithmetic, the
whole expression compiles down to a single rodata load (for the
indices, if used) plus an SIMD reduction. When the lambda is
`|i| i*i`, the compiler can constant-fold the entire tensor into
`.rodata` and emit a single `lea` plus the reduction.

The same shape works on ranges:

```esque
+/(0..5)        // 10
+/(1..=5)       // 15
```

## Combining pipelines and reductions

```esque
fn rms[N](x: f32[N]) -> f32 = {
    let n  = N as f32;            // hypothetical: shape values as scalars
    let sq = x .* x;
    +/(sq) / n
}
```

(`N as f32` is a small convenience that has not yet landed; see the
[roadmap](/reference/planned/roadmap). Use a separate `len`
parameter for now.)

You can usually arrange numeric code so the body of a function is
one pipeline ending in a reduction. That reads well and tends to
compile well.

## Next: [Loop primitives](loop-primitives)
