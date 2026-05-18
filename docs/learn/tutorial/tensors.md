---
title: Tensors
---

# Tensors

A *tensor* in esque is a rectangular block of values of one element
type, with a known shape. esque is designed around tensors the way
Rust is designed around ownership: they are not a library, they are
the language.

## Tensor types

`f32[3]` is a rank-1 tensor of three `f32`s. `f32[3, 4]` is a rank-2
tensor (three rows, four columns each). `i32[N]` is a rank-1 tensor
of `i32`s with shape parameter `N`.

```esque
let v: f32[3]    = [1.0, 2.0, 3.0];
let m: f32[2, 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
```

Shapes are part of the type. `f32[3]` and `f32[4]` are different
types. The compiler will not let you confuse them.

## Tensor literals

A bracketed list of values is a tensor literal:

```esque
[1.0, 2.0, 3.0]              # f32[3]
[1, 2, 3, 4]                 # i32[4]
[[1.0, 2.0], [3.0, 4.0]]     # f32[2, 2]
```

Constant tensor literals are placed in `.rodata` at link time and
loaded with a single `lea`. There is no per-element store at runtime.

## Element-wise arithmetic

Each scalar arithmetic operator has a "dotted" tensor counterpart:

| Scalar | Tensor      | Effect                                |
|--------|-------------|---------------------------------------|
| `+`    | `.+`        | element-wise add                      |
| `-`    | `.-`        | element-wise subtract                 |
| `*`    | `.*`        | element-wise multiply (Hadamard)      |
| `/`    | `./`        | element-wise divide                   |

```esque
let a = [1.0, 2.0, 3.0];
let b = [4.0, 5.0, 6.0];
a .+ b                    # [5.0, 7.0, 9.0]
a .* b                    # [4.0, 10.0, 18.0]
```

Both operands must have the same type, including shape. Broadcasting
between mismatched shapes is [planned](/reference/planned/roadmap),
not implemented.

## Reductions

`+/`, `*/`, etc. reduce a tensor to a scalar by folding the operator
across the elements:

```esque
let a = [1.0, 2.0, 3.0];
let s = +/(a);            # 1.0 + 2.0 + 3.0 = 6.0
let p = */(a);            # 1.0 * 2.0 * 3.0 = 6.0
let dot = +/(a .* a);     # sum of squares
```

A reduction is just an operator with `/` after it. It works for any
operator the type supports.

## A real example: dot product

```esque
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)

fn main() -> i32 = {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = dot(a, b);     # 32.0
    result as i32                # exit 32
}
```

That is the entire dot-product implementation. The compiler picks an
SIMD reduction strategy based on `N` (AVX2 for multiples of 8, SSE
for multiples of 4, with a scalar tail otherwise). You write the
expression; the compiler picks the lanes.

## Building tensors with `tabulate`

```esque
tabulate(5, |i| i * i)    # [0, 1, 4, 9, 16]
```

`tabulate(N, f)` calls `f(0)`, `f(1)`, … `f(N-1)` and packs the
results into a rank-1 tensor of length `N`. It is the index-driven
counterpart of `[..., ..., ...]`.

`tabulate` is currently unrolled at compile time, so `N` must be a
literal `≤ 32`. Larger `N` falls under the planned
[`OpTabulateLoop`](/reference/planned/roadmap).

## Index ranges as tensors

```esque
0..5      # i32[5] = [0, 1, 2, 3, 4]
1..=5     # i32[5] = [1, 2, 3, 4, 5]
```

A range is just a tensor. You can pipe it, reduce it, map it:

```esque
+/(0..5)              # 10
+/(tabulate(5, |i| i * 2))   # 20
```

Today both bounds must be literal integers and the size must be
≤ 1<<20 elements. Dynamic-bound ranges remain
[planned](/reference/planned/roadmap).

## Casts

A scalar `as` cast applies element-wise when used inside a `tabulate`
or after a reduction:

```esque
let dot_f = +/(a .* b);
dot_f as i32              # truncate to int (exit code, etc.)
```

A whole-tensor `as` (e.g. `f32[3] as i32[3]`) is not yet implemented;
build the cast tensor element-wise inside `tabulate`.

## Next: [Pipelines and reductions](pipelines-and-reductions)
