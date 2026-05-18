---
title: Traits
---

# Traits

**Target version:** v0.14.

## What

Static, dispatched-at-monomorphisation typeclass-like traits.

```esque
trait Numeric {
    const ZERO: Self
    const ONE:  Self
    fn add(a: Self, b: Self) -> Self
    fn mul(a: Self, b: Self) -> Self
}

impl Numeric for f32 {
    const ZERO: f32 = 0.0
    const ONE:  f32 = 1.0
    fn add(a: f32, b: f32) -> f32 = a + b
    fn mul(a: f32, b: f32) -> f32 = a * b
}
```

A function may take a generic with a trait bound:

```esque
fn sum[T: Numeric, N](v: T[N]) -> T = +/(v)   # works for any Numeric T
```

## Why not today

esque has shape generics, not type generics. To add traits the
type checker needs:

- Element-type generic parameters (the `[T]` part).
- A trait registry, with subtype-of-trait reasoning.
- Monomorphisation that emits one specialised copy per
  `(shape, trait-impl)` pair.

This is a clean feature but not a small one.

## Why monomorphisation, not vtables

Two reasons. First, traits will be used in the hot path (numeric
generics for autodiff and for the kernel DSL); a vtable indirection
defeats vectorisation. Second, the rest of the language is
already monomorphised on shape; type monomorphisation is the same
pass.

A `dyn Trait` form for host-only data is a possible later
addition, but not part of the v0.14 work.
