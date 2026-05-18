---
title: Scan / prefix sum
---

# Scan / prefix sum

```esque
# prefix.esq
# scan(0, |a, x| a+x, [1,2,3,4]) = [1, 3, 6, 10]; +/ = 20
fn main() -> i32 = +/(scan(0, |a, x| a + x, [1, 2, 3, 4]))
```

```bash
$ ./esquec build prefix.esq -o prefix
$ ./prefix; echo $?
20
```

## What scan does

Given:

- `init: T`
- `f: (T, T) -> T`
- `v: T[N]`

`scan(init, f, v)` returns a `T[N]` where element `i` is
`f(... f(f(init, v[0]), v[1]) ..., v[i])`. It is the prefix
accumulator pattern.

For our input:

| i | running acc          | output[i] |
|---|----------------------|-----------|
| 0 | 0 + 1 = 1            | 1         |
| 1 | 1 + 2 = 3            | 3         |
| 2 | 3 + 3 = 6            | 6         |
| 3 | 6 + 4 = 10           | 10        |

Sum: `1 + 3 + 6 + 10 = 20`.

## Why this is more useful than it looks

A surprising number of numeric problems are scans:

- Cumulative sums (any "running total").
- Cumulative max / min (running peak).
- Histogram-style cumulative counts.
- Some sequence transformations in DP and parsing.

If you can frame your problem as a scan, you typically also unlock
parallelism — work-efficient parallel scan is a well-studied pattern.
esque's `scan` is unrolled at compile time today (`N ≤ 32`), but the
operator is the same one a parallel implementation would use.

## A second example: running max

```esque
fn main() -> i32 = {
    let v = [3, 1, 4, 1, 5, 9, 2, 6];
    let m = scan(0, |a, x| if x > a { x } else { a }, v);
    +/(m)
}
```

`m = [3, 3, 4, 4, 5, 9, 9, 9]`, sum = `46`.
