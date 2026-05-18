---
title: Tensors
---

# Spec: tensors

## Tensor values

A tensor is a finite, rectangular block of values of a single
element type, with a shape known at type-check time after
monomorphisation.

## Layout

The default layout is dense, row-major: for `T[d_1, ..., d_r]` the
element at logical index `(i_1, ..., i_r)` is at byte offset
`(((i_1 * d_2 + i_2) * d_3 + ...) + i_r) * sizeof(T)`.

Alternative layouts (column-major, blocked, strided, sparse) are
**(planned)**.

## Construction

| Form                       | Effect                                      |
|----------------------------|---------------------------------------------|
| `[v_0, v_1, ..., v_{N-1}]` | Rank-1 tensor of length `N`.                |
| `[[v00, v01], [v10, v11]]` | Rank-2 tensor (must be rectangular).        |
| `lo..hi`, `lo..=hi`        | Rank-1 i32 range.                           |
| `tabulate(N, f)`           | Rank-1 tensor from `f(0)…f(N-1)`.           |

Constant tensor literals must be placed in `.rodata` (or its target
equivalent) by the implementation; per-element store at runtime is
not required.

## Operations

### Element-wise

`a .+ b`, `a .- b`, `a .* b`, `a ./ b`, `a .% b` — both operands
must have equal type (element type and full shape). Result has the
same type. Broadcasting is **(planned)**.

### Reductions

`<op>/(v)` — left-fold `op` across `v`'s elements; result is a
scalar of `v`'s element type. Reductions over rank > 1 today reduce
all elements; axis-aware reductions are **(planned)**.

### Matrix multiply **(parsed, codegen planned)**

`a @ b` is reserved for matmul. The expected typing is
`f32[M, K] @ f32[K, N] -> f32[M, N]`. The operator parses today
but does not yet codegen.

### Indexing **(planned)**

A tensor element-access syntax (`a[i]`, `a[i, j]`, slicing,
gather) is **(planned)**. Today the loop primitives consume tensor
elements via `OpTensorElem` internally, but no surface syntax
exposes a single element. For debugging, `each(v, print_i32)`
walks a tensor.

## Lifetime

Every tensor today is either constant (in `.rodata`) or
stack-allocated. There is no heap allocator. Dynamic shapes and
heap-allocated tensors arrive with the
[planned linear types](/reference/planned/linear-types) work.

## Device qualifiers **(planned)**

A tensor type may carry a device qualifier (`on host`, `on device`,
`on shared`, `on unified`) once the
[GPU backend](/reference/planned/gpu-backend) lands. There are no
device qualifiers today; all tensors are implicitly host-resident.
