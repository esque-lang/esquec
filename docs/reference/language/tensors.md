---
title: Tensors
---

# Tensors

A tensor is a rectangular block of values of one element type, with
a shape known at type-check time. esque tensors are first-class
values.

## Shape grammar

```
TensorType  = ElemType "[" Shape "]"
Shape       = ShapeExpr ("," ShapeExpr)*
ShapeExpr   = ShapeTerm (("+" | "-") ShapeTerm)*
ShapeTerm   = ShapeFactor (("*" | "/") ShapeFactor)*
ShapeFactor = nat_literal | shape_var | "(" ShapeExpr ")"
```

Shape arithmetic in type position is parsed today. Evaluation of
non-trivial shape expressions awaits the planned const-eval work.

## Tensor literals

```
[1.0, 2.0, 3.0]                       // f32[3]
[1, 2, 3, 4]                          // i32[4]
[[1.0, 2.0], [3.0, 4.0]]              // f32[2, 2]
```

The element type is inferred from the first element. The shape is
inferred from the literal's structure. Non-rectangular nested
literals are a type error.

## Range expressions

```
0..5         // i32[5] = [0, 1, 2, 3, 4]   (exclusive)
1..=5        // i32[5] = [1, 2, 3, 4, 5]   (inclusive)
```

Both bounds must be integer literals today. The element type is
`i32`. The size is capped at 1 << 20 to avoid runaway `.rodata`
growth.

## Element-wise operators

| Op   | Effect                                       |
|------|----------------------------------------------|
| `.+` | element-wise add                             |
| `.-` | element-wise sub                             |
| `.*` | element-wise multiply (Hadamard)             |
| `./` | element-wise divide                          |
| `.%` | element-wise modulo                          |

Both operands must have the same type, including shape. There is no
broadcasting today; matched shapes only.

## Reductions

`<op>/(v)` folds `op` across `v`'s elements:

```
+/(v)    // sum
*/(v)    // product
-/(v)    // running diff (rare; usually use scan)
```

Reductions take a tensor and return a scalar of the element type.
Reductions on rank-2+ tensors reduce all elements (not "along an
axis"); axis-aware reductions are planned.

### Codegen for `+/`

| Element type | Shape | Strategy                        |
|--------------|-------|---------------------------------|
| `f32`        | mult of 8 | AVX2 `vaddps` + `vhaddps`   |
| `f32`        | mult of 4 | SSE `addps` + SSE3 `haddps` |
| `f32`        | irregular | `[8] + [4] + scalar tail`   |
| `i32`        | any   | scalar add chain                |

## Matrix multiply

```
a @ b
```

`a: f32[M, K]`, `b: f32[K, N]`, result `f32[M, N]`. The token is
parsed at precedence 35, but the codegen path is not yet wired.
Until it is, write the matmul explicitly with `tabulate`/`+/`.

## Indexing

There is no element-access syntax today. To get one element, use
the loop primitives or destructure (when records arrive). For
debugging, `each(v, print_i32)` walks a tensor.

## Stride / layout

Tensors are densely packed in row-major order in memory.
Constant tensors live in `.rodata` aligned to 16 bytes (so the SIMD
loads are aligned). Local tensors live in stack slots.

## Lifetime

All tensors today are stack- or rodata-bound. Heap allocation
arrives with dynamic shape support.
