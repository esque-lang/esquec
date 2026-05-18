---
title: Types
---

# Spec: types

## Kinds

The type system has two kinds:

- `*` — value types (every type that classifies values).
- `nat` — non-negative integer values usable in shapes.

There is no kind polymorphism. Shape parameters live in `nat`.

## Primitive types

| Type    | Kind | Codegen status                     |
|---------|------|------------------------------------|
| `i8`    | `*`  | (impl, v0.12)                      |
| `i16`   | `*`  | type-checks, **planned** codegen   |
| `i32`   | `*`  | (impl)                             |
| `i64`   | `*`  | (impl, v0.11)                      |
| `u8`    | `*`  | (impl, v0.12)                      |
| `u16`   | `*`  | type-checks, **planned** codegen   |
| `u32`   | `*`  | (impl, v0.11)                      |
| `u64`   | `*`  | type-checks, **planned** codegen   |
| `f32`   | `*`  | (impl)                             |
| `f64`   | `*`  | (impl, v0.12)                      |
| `bool`  | `*`  | (impl)                             |
| `unit`  | `*`  | (impl)                             |
| `string`| `*`  | (impl, v0.12; literals only)       |

Implementations must accept the full table; they may emit a
diagnostic when a type is not yet codegened on the target backend.
The `string` type is an immutable UTF-8 fat pointer of `(ptr, len)`;
the full string surface (concatenation, indexing, runtime
construction) is [planned](/reference/planned/strings).

## Tensor types

```
T[d_1, d_2, ..., d_r]
```

`T` is any primitive of kind `*`. Each `d_i` is a *shape dimension*:

- A literal of kind `nat`: `3`, `256`.
- A shape parameter name in scope: `N`, `M`.
- A shape expression of kind `nat`: `N+1`, `2*M`.

The *rank* of a tensor type is the count of dimensions.

### Shape expressions

```
ShapeExpr  = ShapeTerm (("+" | "-") ShapeTerm)*
ShapeTerm  = ShapeFactor (("*" | "/") ShapeFactor)*
ShapeFactor = nat_literal | shape_var | "(" ShapeExpr ")"
```

Shape expressions are **(impl)** at parse, **(planned)** at the
const-eval level: the type checker accepts shape arithmetic in
declaration position, but a binding is established only when the
expression evaluates to a single concrete `nat` after
monomorphisation. Calls that would require non-trivial shape
arithmetic produce a diagnostic citing the planned const-eval
work.

## Type equality

Two types are equal iff:

- Both are the same primitive, **or**
- Both are tensor types with equal element types, equal ranks, and
  equal dimensions in order. Concrete dimensions match by value;
  shape variables match by name (after monomorphisation, all
  dimensions are concrete).

## Function types

A function has a type of the form

```
fn[shape_params](param_types) -> ret_type
```

Function values are not first-class today; lambdas are inlined at
use sites, and `each` requires a *named* top-level function (see
[Expressions → Each](expressions)). First-class function types are
**(planned)**, see
[Planned: kernel DSL](/reference/planned/kernel-dsl).

## Casts

The unary postfix `e as T` performs an explicit numeric conversion.
Required cases:

| From | To  | Semantics                  |
|------|-----|----------------------------|
| `i32`| `f32`| convert to nearest f32     |
| `f32`| `i32`| truncate toward zero       |

Other numeric conversions arrive with extended-numerics codegen.

Whole-tensor `e as T'[shape]` is **(planned)**; do the conversion
element-wise via `tabulate` until then.

## Linear and reference types **(planned)**

The full design (linear `!T`, shared borrow `&T`, mutable borrow
`&mut T`) is documented under
[Planned: linear types](/reference/planned/linear-types). esque has
no reference-type machinery today; values are passed by value
(which for tensors means by stack copy, since there is no heap
yet).

## Records and sums **(planned)**

ADTs are **(planned)**; see
[Planned: roadmap](/reference/planned/roadmap).

## Traits **(planned)**

The trait system is **(planned)**. See
[Planned: traits](/reference/planned/traits).
