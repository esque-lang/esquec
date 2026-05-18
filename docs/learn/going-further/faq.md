---
title: FAQ
---

# FAQ

## Why no `for` and `while`?

Because every numeric loop you would write with them is one of a few
named patterns in esque (`tabulate`, `scan`, `iterate_until`,
`each`, `+/`). Naming them makes the program self-explaining and
makes the compiler's job easier. See
[Loop primitives](../tutorial/loop-primitives) for the full pitch.

## Why no LLVM?

Two reasons. First, the compiler is small enough that an LLVM
dependency would be the largest line item in the build. Second, the
backend's job is *narrow* — emit good x86-64 for tensor kernels —
and it benefits from owning isel, regalloc, and encoding end-to-end.
LLVM would actually make some of the work harder.

## Why Linux-only? Why x86-64-only?

Time. The encoder is hand-written, the ELF writer is hand-written,
the syscall stub is hand-written. Adding macOS / Windows / aarch64
is straightforward but real work; nobody has done it yet.

## Where does memory come from?

esque has no heap allocator today. All values are stack-allocated or
in `.rodata`. Tensor literals and constant folds end up in
`.rodata`. Local tensors live in stack slots managed by the
register allocator. Dynamic-shape tensors will need an allocator;
that comes with the [linear types](/reference/planned/linear-types)
work.

## Can I link a C library?

Not yet. The backend does not produce a libc-aware object — the
`_start` stub bypasses libc entirely. FFI is on the long-term
roadmap but is gated on having effect typing first.

## What does `__N` mean in function names?

Monomorphisation. `dot__3` is the specialised copy of `dot[N]` for
`N=3`. The compiler emits one copy per shape it sees at call sites.

## Why does my tensor literal end up in `.rodata`?

Because constant tensors are placed there by the MIR pass, with
appropriate alignment, and referenced from `.text` via
`R_X86_64_PC32`. The single `lea` is faster than per-element
stores on every call.

## What if I want to print a tensor?

`each(v, print_i32)` will print one element per line. There is no
"print this whole tensor as a vector" intrinsic; once general I/O
lands you can build it.

## Where is `string`?

String literals (`"..."`) work as of v0.12 and can be passed to
`print_str`. The full string surface (concatenation, slicing,
length, byte access) is still
[planned](/reference/planned/strings).

## Where is `i64` codegen?

`i32`, `i64`, `u32`, `i8`, `u8`, and `f64` all codegen end-to-end
today. The remaining widths (`i16`, `u16`, `u64`) type-check but
the backend rejects them with a forward-pointing diagnostic; see
[extended numerics](/reference/planned/extended-numerics).

## Where is broadcasting?

Not yet. Element-wise ops require matched shapes today. Broadcasting
is on the roadmap and will likely arrive with the kernel-DSL work.

## Where can I see the spec?

[Reference → Specification](/reference/spec/overview).

## Where is the public roadmap?

[Reference → Planned features → Roadmap](/reference/planned/roadmap).
