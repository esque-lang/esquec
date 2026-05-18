---
title: Printing and I/O
---

# Printing and I/O

esque's I/O surface is intentionally tiny. There are three functions
you can call for their side effect, plus one expression form (`each`)
that drives them across a tensor. Every I/O-performing function — the
intrinsics and any wrappers you write — must be marked `@io` (see
[Effects and I/O](../going-further/effects-and-io)).

## `print_i32(x: i32) -> i32 @io`

Writes `x` to standard output as a decimal integer, followed by a
newline. Returns `x` unchanged so the call can sit in expression
position.

```esque
@io fn main() -> i32 = print_i32(42)
```

```
$ ./hello
42
$ echo $?
42
```

`print_i32` is a *runtime intrinsic*, synthesised by the compiler
when the program references it. It uses the Linux `write` syscall
directly; there is no libc dependency.

## `print_f32(x: f32) -> f32 @io`

Writes `x` as a decimal float (six fractional digits, no trailing
trim today), followed by a newline. Returns `x` unchanged.

```esque
@io fn main() -> i32 = { print_f32(3.14); 0 }
```

```
$ ./hello
3.140000
$ echo $?
0
```

## `print_str(s: string) -> string @io`

Writes the bytes of `s` verbatim. No trailing newline is added.
Returns `s` unchanged.

```esque
@io fn main() -> i32 = { print_str("hello\n"); 0 }
```

## `each(v, f)`

Drive a print across a tensor:

```esque
@io fn main() -> i32 = {
    each(0..5, print_i32);
    0
}
```

```
$ ./prog
0
1
2
3
4
$ echo $?
0
```

`each` accepts any named top-level function whose effect set fits
the enclosing function's. A pure `f` works in any caller; an `@io`
`f` requires the caller to be `@io` too. You can wrap your own
print logic:

```esque
@io fn shout(x: i32) -> i32 = print_i32(x * 100)

@io fn main() -> i32 = {
    each(0..3, shout);
    0
}
```

A closure or captured name is still a type error — `f` must
resolve to a top-level function name.

## What is missing

- Reading input. There is no `read_line`, no `argv`, no `stdin`.
- File I/O.
- Environment / clock / RNG.

These are stdlib gaps, not language gaps. Once we settle on a
runtime story for them they will appear as ordinary `@io`
functions.

## Reading input

There is no input today. Write programs whose inputs are baked in or
computed; treat the exit code or stdout as the output.

## Next: [Worked examples](../examples/)
