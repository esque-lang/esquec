---
title: Performance
---

# Performance

esque is small. The compiler is a single static Go binary; the
output is a single static ELF executable; there is no runtime to
load. That makes performance behaviour easy to predict and easy to
inspect.

## The headline result

`benchmarks/run_benchmarks.sh` compares esque against `gcc -O3
-mavx2` on common kernels. Detailed numbers live in
[`benchmarks/ANALYSIS.md`](https://github.com/esque-lang/esquec/blob/main/benchmarks/ANALYSIS.md);
the summary is "competitive, sometimes faster, on the kernels the
SIMD codegen covers."

## What the backend does well

- **Tensor literals** are placed in `.rodata` and loaded with a
  single `lea`. There is no per-element store cost for constants.
- **Element-wise ops** on `f32[N]` lower to AVX2 `vaddps` /
  `vmulps` (256-bit, 8 lanes) when `N` is a multiple of 8, SSE
  `addps` / `mulps` (128-bit, 4 lanes) when `N` is a multiple of 4
  but not 8, and a hybrid for shapes like 12 or 15.
- **Reductions** (`+/`) emit horizontal-add chains:
  `vhaddps`/`haddps` to collapse SIMD lanes, plus a scalar tail.
- **Constant folding** runs in the CEIR pass, so anything the
  compiler can prove is pure arithmetic of literals becomes a
  single rodata load.

## What it does not yet do

- **Vector tail handling for irregular shapes** is correct but not
  always optimal. A shape like `f32[15]` is `[8] + [4] + [3
  scalar]`; the SSE chunk and the scalar tail are scheduled
  separately. A peephole pass that fuses them is on the roadmap.
- **No autovectorisation across loops.** Because there are no `for`
  loops, this is mostly fine: tensor primitives lower directly to
  SIMD. The exception is unrolled `tabulate(N≤32, ...)`, which can
  produce SIMD-friendly patterns the compiler does not yet detect.
- **No FMA**. `vmulps` + `vaddps` is emitted instead of `vfmadd`.
  The encoder can grow a FMA case; nobody has written it yet.

## How to inspect what the compiler did

```bash
./esquec build foo.esq --emit=ceir | less       # core IR
./esquec build foo.esq --emit=mir  | less       # MIR
./esquec build foo.esq --emit=asm  | less       # hex of emitted code
./esquec build foo.esq --emit=obj -o foo.o
objdump -d foo.o                                 # full disassembly
objdump -s -j .rodata foo.o                      # see the rodata blob
```

The hex output of `--emit=asm` is the actual byte stream the encoder
produced; `objdump` is the friendly view.

## Tactics for fast esque code

1. **Pick shapes that are multiples of 4 or 8.** A multiple of 8
   gets full AVX2 width; a multiple of 4 gets SSE.
2. **Lift constants to module scope** when possible. A bound
   `let v = [...]` inside a hot path is rodata-loaded; the compiler
   does not yet move the literal out.
3. **Use `+/`, not a hand-written fold.** The reduction operator
   takes the SIMD path; a recursive scalar fold does not.
4. **Avoid casts inside reductions.** `as` between `f32` and `i32`
   is cheap but defeats fusion.
5. **Inspect.** When in doubt, `--emit=asm`. The output is short
   enough to read.

## Worst-case behaviour

esque has a deliberately dumb register allocator (linear scan over
MIR). If you write a function with many simultaneously-live SIMD
values it can spill. The `TestFoldAdd16` end-to-end test
specifically exercises the spill path. In practice, the kinds of
programs you write in esque do not hit this — but if `--emit=asm`
shows a lot of `movaps` to/from `[rsp + offset]`, you know what is
happening.
