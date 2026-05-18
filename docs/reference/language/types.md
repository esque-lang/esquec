---
title: Type system
---

# Type system

esque is statically typed with bidirectional inference. Function
signatures are fully annotated; local bindings infer from
initialisers. Tensor shapes are part of types and are checked.

## Kinds

The type system has two kinds:

- `*` — ordinary value types (`i32`, `f32[3]`, …).
- `nat` — shape values (non-negative integers known at compile time
  or at type-check time).

Shape parameters live in `nat`. There is no kind polymorphism today.

## Primitive types

| Type     | Kind | Codegen |
|----------|------|---------|
| `i32`    | `*`  | yes     |
| `i64`    | `*`  | yes (v0.11; scalar) |
| `u32`    | `*`  | yes (v0.11; scalar, unsigned div / cmp) |
| `i8`     | `*`  | yes (v0.12; scalar) |
| `u8`     | `*`  | yes (v0.12; scalar) |
| `i16`, `u16`, `u64` | `*` | type-checks; backend rejects |
| `f32`    | `*`  | yes     |
| `f64`    | `*`  | yes (v0.12; scalar) |
| `bool`   | `*`  | yes     |
| `unit`   | `*`  | yes (zero size) |
| `string` | `*`  | yes (v0.12; immutable UTF-8 fat pointer, literals only) |

The remaining numerics (`i16`, `u16`, `u64`) are recognised by the
parser and type checker but produce a forward-pointing diagnostic
in the current backend; see
[planned: extended numerics](/reference/planned/extended-numerics).

## Tensor types

```
T[d_1, d_2, ..., d_r]
```

`T` is the element type (any primitive of kind `*`). Each `d_i` is a
*shape dimension*, one of:

- A concrete `nat` literal: `3`, `256`.
- A shape parameter name: `N`, `M`, `Batch`.
- A shape expression: `N+1`, `2*M` (parsed; the type checker
  monomorphises before evaluating).

Examples:

```
f32[3]
f32[N]
f32[M, N]
i32[Batch, M, K]
```

The rank is the number of dimensions. There is no upper bound today.

## Type equality

Two types are equal iff:

- Both are the same primitive type, or
- Both are tensor types with the same element type, the same rank,
  and equal dimensions in order. Concrete dims match by value;
  variable dims match by name (after monomorphisation, all dims are
  concrete).

```
f32[N, M]  ==  f32[N, M]
f32[N, M]  !=  f32[M, N]
f32[N]     !=  f32[N, 1]
f32[3]     !=  f32[4]
```

## Inference

| Site                    | Inference?                                 |
|-------------------------|--------------------------------------------|
| Function parameter      | No — must be explicit.                     |
| Function return type    | Required today.                            |
| `let x = e`             | Inferred from `e`.                         |
| `let x: T = e`          | `T` checked against `e`'s type.            |
| Tensor literal element type | Inferred from the first element.       |
| Tensor literal shape    | Inferred from the literal length.          |
| Lambda parameter type   | Propagated from the call context.          |
| Lambda return type      | Inferred from the body.                    |
| Shape parameter binding | Inferred at the call site (monomorphisation). |

## Casts (`as`)

`e as T` performs an explicit numeric conversion. Defined casts:

| From | To  | Effect                                |
|------|-----|---------------------------------------|
| `i32`| `f32` | int → float (loses precision past 2^24) |
| `f32`| `i32` | float → int (truncates toward zero)   |

Whole-tensor casts (`f32[N] as i32[N]`) are not yet implemented; do
the cast element-wise inside `tabulate` or before the literal.

## Operator typings

| Op                     | Operands                          | Result |
|------------------------|-----------------------------------|--------|
| `+ - * / %`            | scalar `T, T`, T numeric          | `T`    |
| `.+ .- .* ./ .%`       | tensor `T[shape], T[shape]`       | `T[shape]` |
| `<op>/`                | tensor `T[shape]`                 | `T`    |
| `== != < <= > >=`      | `T, T` for any equality-supporting `T` | `bool` |
| `&& \|\|`              | `bool, bool`                      | `bool` |
| `!`                    | `bool`                            | `bool` |
| `\|>`                  | desugar to call                   | (call result) |

`-` (unary) on `T` returns `T`. `!` on `bool` returns `bool`.

## Functions and shape generics

```
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)
```

Shape parameters in `[ ]` are kind `nat`. Calls infer the shape
parameter binding from the actual operands and the type checker
emits one specialised copy per binding seen at call sites
(monomorphisation). The naming convention is `name__d1_d2_…`.

## Built-in primitives (typing)

The loop primitives have these signatures (informal):

```
ranges:        i32[N]                    where N = hi - lo (or +1 for ..=)

tabulate(N, f: i32 -> T)        -> T[N]
scan(init: T, f: (T,T) -> T, v: T[N])    -> T[N]
iterate_until(init: T, step: T->T, pred: T->bool, max: i32)  -> T
each(v: T[N], f: T -> unit)              -> unit
iterate(n: i32, init: T, f: T -> T)      -> T
```

`f` argument to `each` must be a named top-level function whose
[effect set](/reference/planned/effects) is a subset of the
enclosing function's (see [Loop primitives](loop-primitives)). As of
v0.13 any `@io` function — `print_i32`, `print_f32`, `print_str`,
or a user-defined wrapper — is accepted; the pre-v0.13 hardcoded
allowlist is gone.

## Unit values

The `unit` type has a single value. It is the type of a block whose
final form is a statement (`expr;`) and the result of `each`. Today
unit values are not first-class beyond return positions; you cannot
bind them with `let _ = …`. (You can use a regular discard inside a
block.)
