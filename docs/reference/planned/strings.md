---
title: Strings and bytes
---

# Strings and bytes

**Status:** literal subset shipped in v0.12; full surface still
pending.

## What

A string type. Concretely: an immutable, length-prefixed UTF-8
sequence of bytes, with literal syntax `"..."`. Plus a `bytes` type
for raw, non-textual byte arrays.

```esque
let greeting: string = "hello"
let n: i32 = greeting.len()
```

## Already in v0.12

- `"..."` literals parse and type-check as `string`.
- A string value is held as a fat pointer `(ptr, len)`; the bytes
  live in `.rodata`. CEIR represents the value as a single
  pointer SSA Value and tracks the constant length in a side
  table that the MIR lowering and the call site consult to
  reconstitute the second ABI register.
- One consumer: `print_str(s: string) -> string`. The runtime
  intrinsic shuffles `(RDI, RSI)` into the System V `write(2)`
  syscall convention, returns the same fat pointer in
  `(RAX, RDX)`.
- Forward-pointing diagnostics reject string concatenation,
  string-typed function parameters, and string-typed function
  returns from non-runtime functions.

## Why the rest is not yet today

Strings are entangled with three other things:

1. **An allocator.** Constant strings can live in `.rodata`, but
   `string + string` produces a new buffer that has to come from
   somewhere. Today there is no heap.
2. **Effect typing.** `string` becomes useful primarily through
   `print(s: string) -> unit @io`. Without `@io` the print is
   awkward (special-cased in `each`).
3. **A real character story.** UTF-8 vs codepoint indexing,
   normalisation, escapes — these are easy decisions to make
   wrong.

## Surface design (sketch)

| Form                | Effect                                         |
|---------------------|------------------------------------------------|
| `"hello"`           | string literal; in `.rodata`                   |
| `s.len()`           | length in bytes                                |
| `s.byte(i)`         | i'th byte as `u8`                              |
| `s ++ t`            | concatenation (heap-allocates; `@alloc` effect)|
| `bytes_of(s)`       | view as `bytes`                                |
| `bytes[]`           | distinct from `string` for non-UTF-8 data      |

## Bytes

`bytes` is a tensor-shaped sequence of `u8`. It exists primarily
for I/O paths where the data is not necessarily textual: file
reads, network frames, etc.
