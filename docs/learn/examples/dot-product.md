---
title: Dot product
---

# Dot product

```esque
# dot.esq
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)

fn main() -> i32 = {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = dot(a, b);     # 32.0
    result as i32                # exit 32
}
```

```bash
$ ./esquec build dot.esq -o dot
$ ./dot; echo $?
32
```

## Notes

- `[N]` is a shape parameter. `dot` is generic over the length of
  its operands; both arguments must have the same shape.
- `.*` is element-wise multiplication: `[1*4, 2*5, 3*6] = [4,10,18]`.
- `+/` reduces the resulting tensor by addition: `4+10+18 = 32`.
- The cast `result as i32` truncates toward zero; we use it because
  exit codes are integers.

## Monomorphisation

The call `dot(a, b)` where both operands have shape `[3]` causes the
compiler to emit a specialised copy `dot__3`. Add a second use:

```esque
fn use4(p: f32[4], q: f32[4]) -> f32 = dot(p, q)
```

…and the compiler also emits `dot__4`. `dot__3` and `dot__4` are
independently scheduled and (for SIMD-aligned shapes) use different
backends — `dot__4` and `dot__8` are full SSE/AVX2 paths; `dot__3`
takes the scalar tail.

You can see all of this with `--emit=ceir`:

```bash
$ ./esquec build dot.esq --emit=ceir | head
```

Look for the function name `dot__3` (it will not say plain `dot`).

## Generalising

A shape-generic `rms` (root mean square) is one extra line:

```esque
fn rms[N](x: f32[N]) -> f32 = {
    let sq = x .* x;
    +/(sq) / 3.0          # hardcoded N=3 until shape-as-value lands
}
```

Until shape values can flow as runtime scalars (planned), pass `N`
as a separate `f32` parameter or use a fixed shape.
