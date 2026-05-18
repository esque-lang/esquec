---
title: Effects and I/O
---

# Effects and I/O

esque tracks side effects in the type system. As of **v0.13** the
only effect is `@io` — a function that may write to stdout (or, in
future, to files / the environment). Pure functions (no annotation)
may not call `@io` functions; `@io` functions may call anything.

## What esque has today

- `print_i32(x: i32) -> i32 @io` — write a decimal integer + `\n`,
  returning `x`.
- `print_f32(x: f32) -> f32 @io` — write a decimal float + `\n`,
  returning `x`.
- `print_str(s: string) -> string @io` — write a string verbatim,
  returning `s`.
- `each(v, f)` where `f` is any named function whose effect set is
  a subset of the enclosing function's effect set.

Everything else is pure. There is no file I/O, no environment
access, no clock, no randomness, no general stdout writer — but
adding any of those is now a stdlib question, not a language one.

## The rule

```
callee.effects ⊆ caller.effects
```

A pure caller may call only pure callees. An `@io` caller may call
either pure or `@io` callees. `each(v, f)` applies the same rule to
its second argument.

```esque
@io fn shout(x: i32) -> i32 = print_i32(x)

@io fn main() -> i32 = {
    each(0..3, shout);    # ok: shout is @io, main is @io
    0
}

fn pure_main() -> i32 = print_i32(1)   # type error
```

The type error names both sides and suggests adding the annotation
to the caller.

## Why the design

esque is built around the idea that the type system should tell you
what code can do. A function annotated `(i32) -> i32` cannot
secretly produce side effects. `each` does not have to trust an
arbitrary function; it just checks the effect set fits.

## What you write today

- Mark `main` as `@io fn main()` whenever you want to print.
- Wrap repeated print logic in your own `@io fn`. The pre-v0.13
  hardcoded allowlist (`print_i32`, `print_f32`) is gone — any
  effect-correct function works.
- Pure-by-default functions stay easy to read: if there is no
  annotation, there is no I/O.

## Beyond `@io`

Other effects on the long-term roadmap:

- `@alloc` — allocates heap memory (relevant once dynamic-shape
  tensors arrive).
- `@panic` — may abort.
- `@nondet` — random / nondeterministic.

These are all ways for the type system to keep promising more
about what a function does. The `EffectSet` representation is
already a bitmask, so adding each new effect is a one-bit change
plus its propagation rules.
