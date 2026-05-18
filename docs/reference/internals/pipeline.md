---
title: Pipeline overview
---

# Pipeline overview

```
.esq source
  ↓ internal/lex          hand-written byte-level UTF-8 lexer
  ↓ internal/parse        recursive descent + Pratt
  ↓ internal/ast          AST node types
  ↓ internal/types        two-pass: collect signatures, then check bodies
  ↓ internal/ceir         core IR (ANF, SSA, fully typed); ConstFold pass
  ↓ internal/mir          matrix IR; near-identical to CEIR today
  ↓ internal/backend/cpu/x86   isel + regalloc + encoding
  ↓ internal/object/elf   pure-Go ELF64 relocatable object writer
  ↓ internal/link         shells out to system `ld` with `-e _start`
ELF executable
```

Each `--emit=` flag short-circuits the driver at the corresponding
boundary; see [CLI](../../going-further/cli) (Learn track).

## Other internal packages

- **`internal/diag`** — structured `*diag.Diagnostic` errors with
  `Span` / `Pos` and a `Render(src)` method that draws a caret. The
  driver's `reportErr` uses `errors.As` to fall through to plain
  `%v` for non-diagnostic errors. New error sites should produce
  `*diag.Diagnostic` rather than `fmt.Errorf`.
- **`internal/hir`** — currently near-empty; reserved for an
  AST→HIR desugaring pass (pipeline, reductions, broadcasts) that
  today lives inside `ceir/lower.go`.
- **`internal/autodiff`, `internal/stdlib`** — placeholders.
- **`internal/backend/gpu/{ptx,cuda}`** — early PTX/CUDA backend
  scaffolding; not on the default build path.

## Driver

`cmd/esquec/main.go` wires the stages, owns the `--emit=` flags,
and synthesises runtime intrinsics (`print_i32`, `print_f32`,
`print_str`) on demand by appending hand-written x86-64
implementations into the final object before linking.
