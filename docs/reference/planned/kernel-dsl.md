---
title: Kernel DSL
---

# Kernel DSL

**Target version:** v0.16.

## What

A second surface for writing performance-sensitive numeric code,
with explicit blocking, axis-aware reductions, and per-axis
loops. The kernel DSL is *the* place where you would write a
matmul or a stencil by hand.

```esque
kernel matmul[M, K, N](
    a: f32[M, K], b: f32[K, N]
) -> f32[M, N] {
    // tile the loops
    for i in tile(M, 32) {
        for j in tile(N, 32) {
            for k in tile(K, 8) {
                // inner block uses element-wise on f32[32, 32], f32[32, 8]
                ...
            }
        }
    }
}
```

The kernel DSL re-introduces `for` — but only inside `kernel`
declarations, where its semantics are tight loops over a
compile-time-known iteration space. Pure-language code outside a
kernel keeps the loop primitives.

## Why a separate surface

esque's main pitch is that programs read top-down as data
transformations. That is exactly the opposite of how a fast
matmul is written: a matmul *is* its loop nest, and the
imperative loop reads better than the equivalent fold. Putting
that style behind a `kernel` keyword tells the reader (and the
compiler) which mode they are in.

## Why not today

Two reasons. First, several supporting pieces have to land first
— most importantly the GPU backend (kernels target both CPU and
GPU) and the linear-types story (kernel-local buffers in shared
memory are linear). Second, the design is genuinely hard, and
shipping it before the rest of the language is solid would
constrain it.

## Relationship to existing primitives

The pure-language loop primitives (`tabulate`, `scan`,
`iterate_until`, `each`) cover most things you need. The kernel
DSL is for the small fraction of code where those are not enough:
matmul, conv, attention, custom kernels.
