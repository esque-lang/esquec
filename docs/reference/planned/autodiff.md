---
title: Autodiff
---

# Autodiff

**Target version:** v0.17+ (after kernel DSL).

## What

A first-class reverse-mode automatic differentiation pass that
turns an esque function into its gradient. The shape generally
expected:

```esque
fn loss[N](w: f32[N], x: f32[N], y: f32) -> f32 {
    let pred = +/(w .* x)
    let err  = pred - y
    err * err
}

# the @grad attribute requests the compiler synthesise a gradient
@grad(w)
fn dloss_dw[N](w: f32[N], x: f32[N], y: f32) -> f32[N] = loss(w, x, y)
```

The compiler reads the body of `loss`, runs reverse-mode AD on the
CEIR (which is already SSA / ANF — a good shape for AD), and emits
`dloss_dw` as a real function alongside `loss`.

## Status

`internal/autodiff` exists as a placeholder package. There is no
implementation today. The CEIR-as-source design is the major
prerequisite — AD over a typed SSA IR is well-trodden ground.

## Why not today

Three reasons, in priority order:

1. **Element-type generics first.** Useful AD requires
   [traits](traits) (`Numeric`, `Differentiable`) so the gradient
   carrier is parameterised, not hard-coded to `f32`.
2. **Linear types help.** The reverse pass writes into pre-allocated
   gradient buffers; without [linear types](linear-types), every
   reverse-pass write is a fresh allocation.
3. **Kernel DSL helps.** Hand-written kernels (matmul, conv) are
   exactly the operators that need hand-written VJPs. The AD pass
   wants a registry of "how to differentiate this kernel" and the
   kernel DSL is where those VJP definitions naturally live.

## Sketch

The simplest viable v1:

- `@grad(p)` attribute on a function declaration, where `p` is one of
  the parameter names. Multiple `@grad` attributes are allowed and
  each produces a separate gradient function.
- The AD pass runs on CEIR after type-check, before the const-fold
  pass. It builds a tape (per-op derivative information) then
  unrolls the reverse sweep into ordinary CEIR.
- Standard rules for the existing primitives: `+`, `-`, `*`, `/`,
  element-wise tensor ops, `+/` reductions, `tabulate`, `scan`.
- `iterate_until` differentiates as the unrolled equivalent (it
  already lowers via select-cascade so the reverse sweep is a
  symmetric cascade).
- Pattern match and `if`: standard control-flow AD using a
  per-branch tape.

## Long term

- Forward-mode (`@jvp`) for use cases where the input is small and
  the output is large.
- Higher-order derivatives via repeated `@grad` (the pass is
  idempotent if applied to its own output).
- Cotangent-passing-style for memory-efficient checkpointing on
  long `iterate_until` loops, once the runtime-loop form lands.
