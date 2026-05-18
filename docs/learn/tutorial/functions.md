---
title: Functions
---

# Functions

A function in esque is a name bound to a single expression.

```esque
fn add(a: i32, b: i32) -> i32 = a + b
```

That is the whole declaration: `fn name(params) -> ReturnType = body`.
There is no `return` *statement* — the body **is** the result.
(`return` is a reserved word but has no use today.)

## Multi-step bodies

Use a block to compose several `let` bindings before the final
expression:

```esque
fn sum_of_squares(a: i32, b: i32) -> i32 = {
    let sq_a = a * a;
    let sq_b = b * b;
    sq_a + sq_b
}
```

Semicolons separate steps. The final expression has no trailing
semicolon — it is the return value.

## Calls

```esque
fn double(x: i32) -> i32 = x * 2
fn main() -> i32 = double(21)
```

Order of arguments matters. Calls are eager and left-to-right.

## Recursion

```esque
fn factorial(n: i32) -> i32 = {
    if n <= 1 { 1 }
    else { n * factorial(n - 1) }
}

fn main() -> i32 = factorial(5)   // 120
```

esque is fine with self-recursion and mutual recursion. The compiler
collects all top-level signatures in a first pass before checking any
body, so order of definitions does not matter.

## Lambdas

`|params| body` is an anonymous function. Lambdas appear only as
inline arguments to higher-order primitives like `tabulate`, `scan`,
and `iterate_until`:

```esque
fn main() -> i32 = +/(tabulate(5, |i| i*i))   // 0+1+4+9+16 = 30
```

Binding a lambda to a `let` and calling it later is **not** supported
today — first-class function values are planned. Lambdas capture
nothing (no closures over outer state). They exist to inline
computation into a generic; the compiler specialises and unrolls
them away.

## Pipeline calls

`x |> f` is exactly `f(x)`:

```esque
fn double(x: i32) -> i32 = x * 2
fn add_one(x: i32) -> i32 = x + 1
fn square(x: i32) -> i32 = x * x

fn main() -> i32 = 3 |> double |> add_one |> square
//                  = square(add_one(double(3))) = square(7) = 49
```

`x |> f(y)` is `f(x, y)`:

```esque
fn add(a: i32, b: i32) -> i32 = a + b
fn main() -> i32 = 10 |> add(5)   // = add(10, 5) = 15
```

Pipelines are *just syntax*. They lower to the same calls you would
write by hand and produce the same machine code.

## Generic over shapes

A function may declare shape parameters in `[ ... ]` between the name
and the parameter list:

```esque
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)
```

`N` is a `nat`-valued shape parameter. Each call site infers `N` from
the actual tensor shape and the compiler emits one specialised copy
per `N` it sees:

```esque
fn use_a(x: f32[4], y: f32[4]) -> f32 = dot(x, y)   // emits dot__4
fn use_b(x: f32[8], y: f32[8]) -> f32 = dot(x, y)   // emits dot__8
```

Shape parameters can take an explicit kind: `[N: nat]` is the same as
`[N]`. Multiple shape parameters are comma-separated:

```esque
fn matshape[M, K, N](
    a: f32[M, K], b: f32[K, N]
) -> f32 = 0.0
```

We come back to shapes in detail in [Tensors](tensors).

## Next: [Tensors](tensors)
