---
title: Intrinsics
---

# Intrinsics

The compiler synthesises a small set of runtime intrinsic functions
on demand: if a compiled module has an unresolved relocation to one
of these names, *and* the user did not define a function with the
same name, the driver appends a hand-written x86-64 implementation
to the program before linking.

All print intrinsics carry the `@io` effect, so any function that
calls one (directly or transitively) must itself be annotated
`@io`. A pure function that calls `print_i32` is a type error; see
[Effects (`@io`)](/reference/planned/effects) for the propagation
rules.

## `print_i32`

```
@io fn print_i32(x: i32) -> i32
```

Writes `x` to stdout as a base-10 signed integer, followed by a
newline. Returns `x` unchanged so the call can sit in expression
position (`let y = print_i32(compute()) + 1`).

**Implementation**: an inlined decimal-formatter and a Linux
`write(2)` syscall (no libc).

**Source**: `internal/backend/cpu/x86/runtime.go`,
`BuildPrintI32` (registered via `IsPrintBuiltin`).

## `print_f32`

```
@io fn print_f32(x: f32) -> f32
```

Writes `x` to stdout as a fixed-point decimal with six fractional
digits, followed by a newline. Returns `x` unchanged.

**Caveats**: no NaN/Inf handling, no scientific notation, no
trailing-zero trim. This is a debug/tracing intrinsic, not a
production printer.

## `print_str`

```
@io fn print_str(s: string) -> string
```

Writes the bytes of `s` to stdout verbatim. No trailing newline.
Returns `s` unchanged.

## Adding a new intrinsic

The pattern is fixed. To add `print_i64`:

1. Implement `Build_print_i64()` in
   `internal/backend/cpu/x86/runtime.go`.
2. Add the name to `IsPrintBuiltin` so the symbol is recognised as
   a runtime intrinsic and not a missing user function.
3. Add a `case "print_i64":` in `appendRuntimeFns` in
   `cmd/esquec/main.go` to emit the body when needed.
4. Update the type checker to accept `print_i64` as a name with the
   appropriate signature (today these are special-cased; once the
   stdlib package exists they will live there).

There is no other place to touch; the linker picks up the new
symbol automatically.

## What is *not* an intrinsic

- `+/`, `*/`, …: parsed as operators, lowered in CEIR/MIR, codegen'd
  by the SIMD reduction path. Not intrinsics.
- `tabulate`, `scan`, `iterate_until`, `each`, `iterate`: parsed
  specially in the parser; lowered in CEIR. Not intrinsics.
- `as`: parsed as a postfix operator; lowered to MIR cast ops.
