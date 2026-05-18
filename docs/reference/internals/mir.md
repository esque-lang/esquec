---
title: MIR
---

# MIR — Matrix IR

`internal/mir`. The matrix-aware IR sitting between CEIR and the
x86 backend. Today it is *very* close to CEIR — mostly the same set
of ops with light hardware-aware metadata — but it owns a
`RodataEntry` slice that the ELF writer consumes directly.

## Why a separate IR

Three reasons:

1. **A boundary for the kernel-DSL work.** The planned tensor
   kernel DSL needs a layer where matmul-shaped ops, blocking, and
   per-axis reductions are first-class. CEIR is at the wrong level;
   MIR is the layer that grows.
2. **Rodata as data.** MIR carries a `RodataEntry` slice with
   bytes, alignment, and the symbol that `.text` references. The
   x86 backend reads it; the ELF writer reads it.
3. **Sequencing of effects** for the future `@io` work.

## Notable structures

```go
type Module struct {
    Funcs   []*Function
    Rodata  []RodataEntry      // tensor literals destined for .rodata
}

type RodataEntry struct {
    Sym   string         // symbol name in .rodata
    Bytes []byte         // raw payload (little-endian for numerics)
    Align int            // 16 by default; set higher for AVX2
}
```

## Constant scanning

`mir.scanConstants` walks each function and identifies CEIR-side
`OpTensorLit` whose elements are all constants. It serialises the
literal into a `RodataEntry`, replaces the originating
`OpTensorLit` with an `OpRodataPtr` pointing at the new entry,
and the backend then emits a single `lea rip+sym` load.

`flattenTensorLit` handles nested literals (e.g. `[[1.0, 2.0],
[3.0, 4.0]]`) by row-major flatten.

## Where to look

- `internal/mir/mir.go` — types and module shape.
- `internal/mir/lower.go` — CEIR → MIR translation; rodata scan;
  small simplifications.
- `internal/mir/mir_test.go` — assertions on rodata layout for
  range and tabulate fixtures.
