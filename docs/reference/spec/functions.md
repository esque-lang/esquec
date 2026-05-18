---
title: Functions
---

# Spec: functions

## Declarations

```
fn name [shape_params] (params) -> RetType  =  expr

fn name [shape_params] (params) -> RetType { ...; expr }
```

The block form is equivalent to the `=` form with a block
expression as the body.

`shape_params` is an optional `[N, M, ...]` list of identifiers of
kind `nat`. Each may carry an explicit `: nat` annotation.

`params` is a comma-separated list of `name: Type` pairs.

## Purity and effects

Functions are pure by default. A function may not perform I/O,
mutate global state, or read non-`const` globals. Arguments are
immutable.

As of v0.13 a function may opt into the `@io` effect:

```
@io fn main() -> i32 = print_i32(42)
```

`@io` propagates: a caller of an `@io` function must itself be
`@io`. The print intrinsics (`print_i32`, `print_f32`,
`print_str`) are `@io`. See
[Effects and I/O](/reference/planned/effects).

## Lambdas

```
|x| body
|a, x| body
```

Lambdas are anonymous functions used as arguments to higher-order
primitives (`tabulate`, `scan`, `iterate_until`, `iterate`).
Lambdas do not capture surrounding state. They are inlined at use
sites; first-class function values are **(planned)**.

## Calls

```
f(arg, arg, ...)
```

Number of arguments must match the function's parameter count;
each argument's type must match the parameter's type. Calls are
eager and left-to-right.

## Generics over shapes

```
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)
```

Calls infer shape parameter bindings from the actual operands and
the type checker emits one specialised copy per binding seen at
call sites. The naming convention is `name__d1_d2_…` with
shape-parameter values in alphabetical key order.

## First-class function values **(planned)**

A function type `fn(args) -> ret` is **(planned)** as a value form.
Today, lambdas appear only as inline arguments to known
higher-order primitives, and `each` accepts a *named* top-level
function (see [Expressions → Each](expressions)).
