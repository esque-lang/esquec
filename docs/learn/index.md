---
slug: /
title: Welcome to esque
---

# Welcome to esque

**esque** is a statically typed, tensor-primitive systems language. The
compiler `esquec` is written in Go and emits ELF x86-64 Linux executables
directly — there is no LLVM dependency, and no runtime to link.

This site has two parts:

- **Learn** — the path you are on now. A guided tour, then a tutorial that
  teaches the language in the order you will actually meet ideas.
- **[Reference](/reference/)** — the full, terse, definitive description of
  the language and the compiler internals. Useful once you know the basics
  and want to look something up.

If you have never used esque before, you want **Learn**.

## What esque is

esque is built around three observations:

1. **Tensors are not a library; they are values.** A function takes
   `f32[3, N]` the same way it takes `i32`. Shapes are part of the type.
2. **Loops over numeric data are almost always one of a few patterns:**
   build a vector by index, run a prefix accumulator, iterate until a
   condition is met, reduce. esque gives those patterns names
   (`tabulate`, `scan`, `iterate_until`, `+/` …) instead of `for`/`while`.
3. **The compiler should be a single static binary you can read.** No
   build-system rabbit hole; no MLIR, no LLVM, no runtime archive. Just
   parse → typecheck → CEIR → MIR → x86-64 → ELF.

## A 10-second taste

```esque
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)

fn main() -> i32 = {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = dot(a, b);          # 32.0
    result as i32                     # exit code 32
}
```

Compile and run:

```bash
$ ./esquec build hello.esq -o hello
$ ./hello; echo $?
32
```

`+/` is reduce-sum. `.*` is element-wise multiplication. The shape `N`
is inferred at the call site, so `dot([1.0,2.0,3.0], [4.0,5.0,6.0])`
specialises `dot` to `dot__3` automatically.

## Where to next

Brand new? Start with **[Install](install)** then **[Hello, world](hello-world)**.

Already comfortable? Take the **[Tour](tour)**, which fits the whole
language on one page.

Want to look something specific up? Jump to the
**[Reference](/reference/)**.
