---
title: Linear types
---

# Linear types

**Target version:** v0.15.

## What

A small set of reference and ownership annotations on tensor
types, sufficient to support in-place updates and dynamic shapes
without runtime cost:

```
T            // value (small, copyable; for tensors: a handle)
&T           // shared, immutable borrow
&mut T       // exclusive, mutable borrow
!T           // linear: must be consumed exactly once
```

Plus `mut` re-introduced on `let` bindings, and `+=` / `.+=` /
similar mutation operators that re-use the same buffer.

## Why

Without linear types, every element-wise op produces a new buffer.
For small tensors that lives in a stack slot or in `.rodata` and
is fine. For dynamic-shape tensors it forces an allocation per
op, which is a real cost.

Linear types let the compiler see that the LHS of an `a .+= b` is
no longer aliased and the operation can write into `a`'s storage.
The same machinery underpins safe, cheap in-place ops in Rust,
Linear Haskell, and Granule.

## Why not today

Two reasons:

1. **Dynamic shapes come first.** Linear types pay rent only when
   there is a heap to manage. v0.10 has no heap.
2. **Tensor handles are small.** A 24-byte handle (`pointer +
   shape + stride`) is cheap to copy. The pressure for ownership
   tracking shows up when buffers are big and dynamic.

## Sketch of v0.15

- Add `mut`, `&`, `&mut`, `!` to types.
- Track linearity in the type checker (one consumption rule).
- Add `drop(x)` to release a linear handle explicitly.
- Add the in-place ops: `+=`, `.+=`, `.*=`, `./=`.
- Update CEIR and MIR with the borrow-tracking metadata.

The full spec sketch lives in the older spec drafts; see the
[roadmap](roadmap) for milestones.
