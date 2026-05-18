---
title: Extended numerics
---

# Extended numerics

**Target version:** v0.11–v0.12.

## What

Codegen for the integer and float types beyond `i32` and `f32`:

- ~~`i8`~~ *(landed in v0.12)*, `i16`, ~~`i64`~~ *(landed in v0.11)*
- ~~`u8`~~ *(landed in v0.12)*, `u16`,
  ~~`u32`~~ *(landed in v0.11)*, `u64`
- ~~`f64`~~ *(landed in v0.12)*

`i8`, `u8`, `i32`, `i64`, `u32`, `f32`, and `f64` codegen
end-to-end on the v0.12 CPU backend. The remaining widths
(`i16`, `u16`, `u64`) still type-check (the type checker
recognises the suffixes and the type names) but lowering rejects
them with a forward-pointing diagnostic such as

```
error: integer suffix _u16 is parseable but not yet supported by
the v0.12 CPU backend. Other widths are planned for a future
milestone.
```

## Why not today

Three reasons:

1. **Each width is its own encoder work.** Adding 64-bit integer
   ops means new opcodes (REX.W variants) and new register-pair
   handling for some addressing forms. It is mechanical but real.
2. **Unsigned semantics**. `u32` divides differently from `i32`,
   compares differently, and casts differently. Each of these is
   one more case in the type checker.
3. **Tensor element-type generality.** The SIMD codegen for
   reductions and element-wise ops is currently `f32`-specific in
   places. A real `i32` SIMD path is itself a planned task; `i64`
   and `i8` come after that.

## Order of arrival

| Step  | Adds                                              | Status |
|-------|---------------------------------------------------|--------|
| v0.11 | `i64` codegen (literal, scalar; SIMD later)       | done   |
| v0.11 | `u32` + unsigned division and comparison          | done   |
| v0.12 | `f64` codegen                                     | done   |
| v0.12 | `i8` / `u8` codegen                               | done   |
| later | small unsigned types (`u16`, `i16`)               | —      |

## Forward-compatible code today

If you write `42_i32` everywhere you mean `i32`, the day `_i64`
is supported you can simply change the suffix where it matters and
nothing else. The type-suffix syntax was added in v0.10 so users
could pin types now and not have to revisit code later.
