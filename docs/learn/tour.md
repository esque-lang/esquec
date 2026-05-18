---
title: Tour of esque
---

# Tour of esque

This page fits the whole implemented language on one screen. Each section
links into the tutorial for depth.

## Functions

```esque
fn add(a: i32, b: i32) -> i32 = a + b
fn main() -> i32 = add(3, 4)
```

Every `fn` body is a single expression. Use `{ ... }` for a sequence of
`let`-bound steps ending in an expression.

```esque
fn main() -> i32 = {
    let a = 10;
    let b = 3;
    a * b
}
```

→ [Functions](tutorial/functions)

## Types

```esque
i32  i64  f32  f64  bool  unit
f32[3]            // rank-1 tensor of 3 floats
f32[N]            // shape-polymorphic over N
i32[M, K]         // rank-2
```

→ [Values and types](tutorial/values-and-types) · [Tensors](tutorial/tensors)

## Tensor literals and element-wise ops

```esque
let a = [1.0, 2.0, 3.0];   // f32[3]
let b = [4.0, 5.0, 6.0];
let c = a .+ b;            // [5.0, 7.0, 9.0]
let d = a .* b;            // [4.0, 10.0, 18.0]
```

`.+`, `.-`, `.*`, `./` are element-wise. Their non-dotted siblings are
scalar arithmetic.

## Reductions

```esque
+/(a)              // sum of a
+/(a .* b)         // dot product
*/(a)              // product
```

`+/` and `*/` are reductions; the operator before `/` is the combiner.

→ [Pipelines and reductions](tutorial/pipelines-and-reductions)

## Pipelines

```esque
3 |> double |> add_one |> square    // = square(add_one(double(3)))
10 |> add(5)                         // = add(10, 5)
```

`x |> f` is `f(x)`. `x |> f(y)` is `f(x, y)`.

## Generics over shapes

```esque
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)

fn use_a(x: f32[4], y: f32[4]) -> f32 = dot(x, y)   // dot__4
fn use_b(x: f32[8], y: f32[8]) -> f32 = dot(x, y)   // dot__8
```

`[N]` declares a shape parameter. The compiler monomorphises one copy
per concrete shape used at call sites — like Rust generics, but for
shapes.

## Loop primitives (no `for`/`while`)

```esque
0..5         // i32[5] = [0, 1, 2, 3, 4]
1..=5        // i32[5] = [1, 2, 3, 4, 5]

tabulate(5, |i| i*i)                    // [0, 1, 4, 9, 16]
scan(0, |a, x| a + x, [1, 2, 3, 4])     // prefix sums [1, 3, 6, 10]
iterate_until(0, |s| s+1, |s| s == 7, 10)   // 7
each(0..5, print_i32)                    // side effect: prints 0..4
```

→ [Loop primitives](tutorial/loop-primitives)

## Conditionals and pattern matching

```esque
if x > 0 { 1 } else if x < 0 { -1 } else { 0 }

match n {
    0 => 10,
    1 => 20,
    n if n > 100 => 99,
    _ => 0
}
```

→ [Pattern matching](tutorial/pattern-matching) · [Control flow](tutorial/control-flow)

## I/O (today)

```esque
print_i32(42)        // prints 42, returns 42
print_f32(3.14)      // prints 3.14, returns 3.14
print_str("hi\n")    // prints "hi\n", returns "hi\n"
each(v, print_i32)   // prints each i32 element

// callers must be @io to invoke any of the above:
@io fn main() -> i32 = print_i32(42)
```

The print intrinsics return their argument so they fit in expression
position. Effect tracking is via [`@io`](/reference/planned/effects);
a pure function (no annotation) cannot call them.

## Run the tour

Each snippet above is something you can paste into a `fn main()` (or
adapt) and run with `./esquec build foo.esq -o foo && ./foo; echo $?`.

## Next: [Tutorial → Values and types](tutorial/values-and-types)
