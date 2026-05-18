---
title: Values and types
---

# Values and types

esque is statically typed. Every expression has a type, every binding
has a type, and the compiler refuses to compile mismatches.

## Primitive types

| Type   | Description                          | Literal       |
|--------|--------------------------------------|---------------|
| `i32`   | 32-bit signed integer (default int)  | `42`, `'A'`   |
| `i64`   | 64-bit signed integer                | `42_i64`      |
| `u32`   | 32-bit unsigned integer              | `99_u32`      |
| `i8`    | 8-bit signed integer                 | `7_i8`        |
| `u8`    | 8-bit unsigned integer               | `7_u8`        |
| `f32`   | 32-bit float (default float)         | `3.14`        |
| `f64`   | 64-bit float                         | `3.14_f64`    |
| `bool`  | boolean                              | `true`, `false` |
| `string`| immutable byte string (literals only) | `"hi"`       |
| `unit`  | empty value, returned by `each(...)` | (no literal)  |

`i16`, `u16`, and `u64` type-check today but the CPU backend still
rejects them with a forward-pointing diagnostic. They remain on the
[extended-numerics](/reference/planned/extended-numerics) list.

### Integer literals

A bare integer literal like `42` is an untyped constant that
type-checks at the type the context expects (almost always `i32`).
Underscores are allowed for readability:

```esque
let big = 1_000_000;
```

A type suffix forces the type:

```esque
let exit_code = 99_i32;
```

Char literals are i32-typed integer constants holding the Unicode
codepoint:

```esque
fn main() -> i32 = 'A'   # exit 65
```

### Float literals

Float literals always have a decimal point, even when the value is
integral:

```esque
let zero  = 0.0;
let pi    = 3.14159;
```

`3.0`, not `3.`, and never `3` if you mean a float.

### Booleans

```esque
fn main() -> i32 = if true { 1 } else { 0 }
```

`&&`, `||`, `!` work as you would expect. Comparisons (`==`, `!=`,
`<`, `<=`, `>`, `>=`) produce `bool`.

## `let` bindings

```esque
fn main() -> i32 = {
    let x = 10;
    let y = 3;
    x + y
}
```

`let` introduces an immutable binding. There is no `mut`. If you want
to "update" something, write a new binding:

```esque
fn main() -> i32 = {
    let x = 10;
    let x = x + 1;   # shadows; not mutation
    x
}
```

The block `{ ... }` is itself an expression: it evaluates each
semicolon-separated step, then evaluates the final expression and
returns its value.

## Type ascriptions

Bindings infer types from their initialiser, but you can pin one
explicitly:

```esque
let pi: f32 = 3.14;
```

Function parameters always have explicit types. Function return types
are also written explicitly today, though many can be inferred — the
compiler keeps them required for readability.

## Casts

`as` performs explicit numeric casts:

```esque
fn main() -> i32 = {
    let x: f32 = 3.7;
    x as i32                  # truncates toward zero → 3
}
```

There are no implicit numeric coercions. `1 + 1.0` is a type error.

## Tensor types — a preview

```esque
let v: f32[3] = [1.0, 2.0, 3.0];
```

A type like `f32[3]` is a rank-1 tensor of three `f32`s. The shape is
part of the type — `f32[3]` and `f32[4]` are different types and you
cannot pass one where the other is expected. We pick this back up
in [Tensors](tensors).

## Next: [Functions](functions)
