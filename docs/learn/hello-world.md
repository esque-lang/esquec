---
title: Hello, world
---

# Hello, world

esque has string literals and a tiny print surface but no general
I/O, so the "hello world" of the language is **a program that
exits with a number you chose**. That is enough to teach you
everything about the toolchain.

## The shortest program

Save this as `hello.esq`:

```esque
fn main() -> i32 = 42
```

That is the entire program. `main` is a function with zero parameters
that returns `i32` (a 32-bit signed integer). The body of any `fn` in
esque is a single expression introduced by `=`, no `return`, no braces
required.

## Compile and run

```bash
./esquec build hello.esq -o hello
./hello; echo $?
# 42
```

The exit status of the process is whatever `main` returned. That is
the contract.

## Add some arithmetic

```esque
fn main() -> i32 = (10 + 3) * 2 - 1
```

```bash
./esquec build hello.esq -o hello
./hello; echo $?
# 25
```

esque parses arithmetic with the precedence you expect: `*` and `/`
bind tighter than `+` and `-`; parentheses override.

## Print something

If you want stdout output instead of just an exit code, use the
`print_i32` intrinsic. Functions that perform I/O must be annotated
`@io`:

```esque
@io fn main() -> i32 = print_i32(7)
```

```bash
./esquec build hello.esq -o hello
./hello
# 7
echo $?
# 7
```

`print_i32` writes its argument followed by a newline, then returns
its argument unchanged so the call can sit in expression position.
That is why the exit status above is `7`: `main` simply forwards the
return value of `print_i32(7)`. To exit with a different code, sequence
the print and yield a separate value, e.g.
`@io fn main() -> i32 = { print_i32(7); 0 }`.

There is also `print_f32` for floats and `print_str` for strings. The
[`@io` effect](/reference/planned/effects) tracks side effects in
the type system: a pure function (no `@io` annotation) cannot call
an `@io` function.

## Inspect the pipeline

The compiler can dump every intermediate stage:

```bash
./esquec build hello.esq --emit=ast       # parsed AST
./esquec build hello.esq --emit=ceir      # core IR (SSA, ANF, fully typed)
./esquec build hello.esq --emit=mir       # matrix IR
./esquec build hello.esq --emit=asm       # hex of emitted machine code per fn
./esquec build hello.esq --emit=obj -o hello.o   # stop at the .o file
```

These are explained in the
[Reference → Compiler internals](/reference/internals/pipeline).

## Next: [Tour of esque](tour)
