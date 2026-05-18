---
title: Specification overview
---

# Specification: overview

This is the language specification for esque. It describes what
the language *is*, independent of any one implementation. Where
features are implemented in the reference compiler `esquec` they
are marked **(impl)**; where they are planned they are marked
**(planned)** with a link to the relevant
[Planned features](/reference/planned/overview) page.

The spec was rewritten in v0.10 to remove C-style `for` and `while`
loops, comprehensions in their old form, and the `mut` family of
mutating operators. The replacement primitives — ranges, `tabulate`,
`scan`, `iterate_until`, `each`, and the reduction operator
`<op>/` — are laid out in [Expressions](expressions).

## What esque is

esque is:

- a statically typed, expression-oriented language;
- with tensors as first-class values, shapes in the type system, and
  shape generics dispatched by monomorphisation;
- compiled ahead of time to native code (today: x86-64 Linux ELF);
- pure by default, with an [`@io` effect](/reference/planned/effects)
  that propagates through call sites (impl as of v0.13).

esque is **not**, and is not trying to become:

- a numerical Python replacement (see Triton, JAX);
- a CUDA replacement (the GPU backend is a planned, secondary
  target);
- a general systems language (memory management is intentionally
  narrow).

## Spec sections

- [Lexical](lexical) — source encoding, tokens.
- [Types](types) — primitives, tensor types, shape parameters.
- [Expressions](expressions) — value forms, control, loop primitives.
- [Functions](functions) — declarations, lambdas, generics.
- [Tensors](tensors) — semantics, layouts, ops.
- [Modules](modules) — file organisation.

## Versioning

Spec versions are aligned with major implementation versions. This
document tracks the **v0.13** language: the core surface laid down
in v0.10, plus extended numerics (v0.11–v0.12), string literals
(v0.12), and `@io` effect typing with a flexible `each` callee
(v0.13).

Feature gates and their target versions live in
[Planned features → roadmap](/reference/planned/roadmap).
