---
title: Exit status
---

# Exit status

The smallest meaningful esque program: return a number from `main`
and let the OS see it as the exit code.

## The program

```esque
// hello.esq
fn main() -> i32 = (10 + 3) * 2 - 1
```

## Run

```bash
$ ./esquec build hello.esq -o hello
$ ./hello; echo $?
25
```

## What is happening

- `fn main() -> i32 = …` declares a zero-argument function returning
  `i32`. This is the entry point.
- `(10 + 3) * 2 - 1` is a single expression. It evaluates to `25`.
- The compiler emits a tiny `_start` that calls `main`, then issues
  the Linux `exit_group` syscall with `main`'s return value.

## Try

- Replace the body with `42`. Re-run. `echo $?` → 42.
- Replace it with `'A'`. (Char literals are i32 codepoints.) → 65.
- Replace it with `99_i32`. (Explicit type suffix.) → 99.
- Replace it with `1 + 1.0`. The compiler refuses to compile it.

## Inspect the build

```bash
$ ./esquec build hello.esq --emit=ceir
fn main() -> i32 {
  %0 = ConstInt 13
  %1 = ConstInt 2
  %2 = Mul %0 %1
  %3 = ConstInt 1
  %4 = Sub %2 %3
  Return %4
}
```

`13` and `26 - 1` have already been computed at the AST level by the
constant-fold pass.

```bash
$ ./esquec build hello.esq --emit=asm
main:
  b8 19 00 00 00       mov eax, 0x19      ; 25
  c3                   ret
```

The whole program is two instructions.
