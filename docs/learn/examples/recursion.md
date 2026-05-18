---
title: Recursion
---

# Recursion

```esque
// factorial.esq
fn factorial(n: i32) -> i32 = {
    if n <= 1 { 1 }
    else { n * factorial(n - 1) }
}

fn main() -> i32 = factorial(5)
```

```bash
$ ./esquec build factorial.esq -o fact
$ ./fact; echo $?
120
```

## Notes

- Functions can call themselves; mutual recursion also works because
  the compiler collects all top-level signatures in a first pass.
- `if` is an expression: both arms produce `i32`, so the whole `if`
  produces `i32`.
- The compiler does **not** yet do tail-call optimisation. Recursive
  programs that recurse deeper than the stack tolerates will
  overflow. Use `iterate_until` or `iterate` for tight numeric loops.

## A second example: GCD

```esque
fn gcd(a: i32, b: i32) -> i32 = {
    if b == 0 { a }
    else { gcd(b, a % b) }
}

fn main() -> i32 = gcd(252, 105)   // 21
```

A textbook Euclidean algorithm. Pattern matching would also work but
the `if`-form reads fine.
