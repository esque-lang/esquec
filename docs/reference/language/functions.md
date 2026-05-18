---
title: Functions
---

# Functions

```
fn name [shape_params] (params) -> RetType  =  expr
```

or

```
fn name [shape_params] (params) -> RetType { ...; expr }
```

The block form is the same expression as the `=` form, just lifted
into a block.

## Components

- **`name`**: identifier; must be unique at module scope.
- **`shape_params`**: optional `[N, M, ...]` list of shape variables
  of kind `nat`. Each may have an explicit `: nat` annotation.
- **`params`**: comma-separated `name: Type` pairs. Param types may
  reference shape parameters.
- **`RetType`**: the return type. Required today.
- **Body**: a single expression. Use a block for multi-step bodies.

## Parameters

```
fn add(a: i32, b: i32) -> i32 = a + b
```

Parameters are positional; there is no keyword-arg syntax. There is
no default value syntax.

## Shape parameters

```
fn dot[N: nat](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)
fn matmul[M, K, N](a: f32[M, K], b: f32[K, N]) -> f32[M, N] = ...
```

Shape parameters are inferred at the call site. The compiler emits
one specialised copy per binding (`dot__3`, `dot__8`, etc.).

## Recursion

Self-recursion and mutual recursion are supported because the type
checker collects all top-level signatures in a first pass before
checking any body.

## Lambdas

```
|x| body
|a, x| body
```

Lambdas are anonymous functions, used as arguments to higher-order
primitives like `tabulate`, `scan`, `iterate_until`, `iterate`.
Lambdas do not capture surrounding state; they are inlined at use
sites and exist primarily so the compiler can specialise the
combiner without a separate top-level definition.

Lambda parameter types are inferred from the call context. The
return type is the type of the body.

## Calls

```
f(a, b, c)
```

The number and types of arguments must match. Calls are eager and
left-to-right.

## Pipeline calls

```
x |> f         // f(x)
x |> f(y)      // f(x, y)
x |> f |> g    // g(f(x))
```

`|>` is left-associative at precedence 1 (the loosest binary
operator — looser than `||`, `&&`, comparisons, and arithmetic).
The piped value always becomes the *first* argument.

## Casts

The `as` operator is a postfix form on expressions:

```
let f: f32 = 3.14;
let i: i32 = f as i32;
```

It is not a function call; it is a special postfix.
