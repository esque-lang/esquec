---
title: Effect system (@io)
---

# Effect system (`@io`)

**Status:** the `@io` effect, propagation through call sites, and
removal of the `each`-allowlist landed in **v0.13**. This page
covers what shipped and what is still ahead.

## What shipped in v0.13

esque is a pure-by-default expression language. Function signatures
carry an `EffectSet` (`internal/types/types.go`); today the only
bit is `EffIO`. The empty set means "pure".

The surface form is an `@`-attribute placed before `fn` — the same
syntax planned for `@kernel`/`@grad`/`@inline`:

```esque
@io fn print_line(s: string) -> i32 = print_str(s)
@io fn caller() -> i32 = print_line("hi")    # @io propagates
fn pure() -> i32 = print_line("hi")          # type error
```

Effects compose: if a function calls any `@io` function its
signature must also be `@io`. The runtime intrinsics
(`print_i32`, `print_f32`, `print_str`) are seeded with `EffIO` in
`registerRuntimeBuiltins`. Both `*ast.Call` and `*ast.Each`
verify `callee.Effects ⊆ caller.Effects`; mismatches produce a
diagnostic that names both sides and suggests adding the
annotation.

The pre-v0.13 hardcoded `{print_i32, print_f32}` allowlist on
`each` is gone — any function whose effects fit the caller is
accepted, including pure user functions and user-defined `@io`
wrappers around the print intrinsics.

## What is still planned

- **Effect-polymorphic core combinators.** `tabulate`, `scan`, and
  `iterate_until` callbacks are pure today. The surface for
  declaring effect-polymorphic combinator inputs is reserved for a
  follow-up milestone.
- **More effect bits.** Other effects on the eventual list:

  | Effect      | Meaning                                          |
  |-------------|--------------------------------------------------|
  | `@io`       | Side effects on the world (stdout, files, ...) — **shipped** |
  | `@alloc`    | Heap allocation                                  |
  | `@panic`    | May abort                                        |
  | `@nondet`   | Reads non-deterministic state (clock, RNG)       |

  These ship one at a time, after `@io`.

## Before vs after

**Before** (v0.12): `each` allowlist. Two intrinsics callable from
plain functions only via `each`. No general `print` discipline.

**After** (v0.13): `print_i32` is a library function with signature
`@io fn print_i32(x: i32) -> i32` (it returns its argument so
calls fit in expression position). `each` accepts any `f` whose
effect is at most the caller's. New intrinsics (file I/O, env
access) can appear in the stdlib without a language change.
