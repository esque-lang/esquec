---
title: Euclidean distance
---

# Euclidean distance

```esque
// dist.esq
fn dist[N](x: f32[N], y: f32[N]) -> f32 = {
    let d  = x .- y;
    let sq = d .* d;
    +/(sq)         // squared distance
}

fn main() -> i32 = {
    let a = [3.0, 4.0];
    let b = [0.0, 0.0];
    dist(a, b) as i32         // exit 25 (sqrt would be 5)
}
```

```bash
$ ./esquec build dist.esq -o dist
$ ./dist; echo $?
25
```

## Notes

- `dist` returns the *squared* distance because there is no `sqrt`
  intrinsic today. (You can implement Newton's method with
  `iterate_until` if you really want one — see the
  [iterate-until example](iterate-until).)
- The body is a chain of three tensor expressions: subtract, square,
  reduce. The compiler emits SIMD for each step independently and
  then fuses where it can.

## Pipeline form

The same function written with `|>`:

```esque
fn dist[N](x: f32[N], y: f32[N]) -> f32 = {
    let d = x .- y;
    d .* d |> sumf
}

fn sumf[N](v: f32[N]) -> f32 = +/(v)
```

For a three-step pipeline this is borderline; `+/(d .* d)` is
shorter. Reach for `|>` when the steps have meaningful names.
