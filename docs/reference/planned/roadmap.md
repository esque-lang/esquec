---
title: Roadmap
---

# Roadmap

A consolidated view of what each upcoming version is meant to add.
Targets are intent, not commitments — earlier rows are firmer than
later ones.

## v0.10

Tensor-first loop primitives, char literals, integer type suffixes.

- Range expressions `lo..hi` / `lo..=hi` (integer literal bounds).
- `tabulate(N, |i| f(i))`, `scan(init, fn, v)`,
  `iterate_until(init, step, pred, max)`, `each(v, fn)`.
- All small-N (≤32) unroll-only.
- `for`, `while`, `break`, `continue` removed from the lexer.
- Character literals (`'a'`, `'\n'`, `'\u{1F600}'`) as `i32`
  codepoints.
- Integer type suffixes (`42_i32`); other widths recognised but
  rejected by codegen with forward-pointing diagnostics.
- Print intrinsics (`print_i32`, `print_f32`) synthesised on demand.
- Tensor literals through the `.rodata` fast path.

## v0.11

Extended scalar numerics, large-N runtime loops, and a const-eval
pass over shape arithmetic. The 64-bit GPR path is wired end-to-end
so literals, arithmetic, and division on `i64` and `u32` produce
correct machine code rather than forward-pointing diagnostics.

- **`i64` scalar codegen.** REX.W-prefixed `mov` / `add` / `sub` /
  `imul` / `idiv` / `cqo` were already in the encoder; the type
  checker no longer rejects `_i64`, and CEIR lowering threads
  `types.I64` through `OpConstInt`. A 64-bit multiply that overflows
  i32 (`1_000_000_i64 * 1_000_000_i64`) gives the right answer.
- **`u32` scalar codegen** with proper unsigned semantics. The
  encoder gained an unsigned `DivReg64` (`F7 /6`) and the unsigned
  condition codes (`CCB`, `CCBE`, `CCA`, `CCAE`). Isel dispatches
  `OpDiv` / `OpMod` on `Type.IsUnsigned()` (XOR RDX, DIV) and
  comparison ops on operand signedness for `< <= > >=`.
- **Large-N runtime-loop forms** for `tabulate`, `scan`,
  `iterate_until`, and `each` via new CEIR ops (`OpTabulateLoop`,
  `OpScanLoop`, `OpIterateUntilLoop`, `OpEachLoop`). The >32-element
  unroll ceiling is gone; loop-aware `lastUse` extension across
  back-edges keeps the linear-scan allocator honest.
- **Const-eval over shape arithmetic.** Range bounds, `tabulate` N,
  and `iterate_until` max accept any expression that folds to a
  constant integer (`tabulate(2*16+1, |i| i)`, `0..(N+1)`). The fold
  supports `+ - * / %` and unary `-`.

See the [overview](overview) for an overall picture of the
implementation today.

## v0.12

Small numerics, strings, and ergonomic let bindings.

- **`let x = e in body`** as a primary expression. Lands first
  because `in` is already a reserved keyword and the form desugars
  to a single-statement `Block` for free. Useful in
  single-expression function bodies and arbitrarily nested. ✓
- **`f64` scalar codegen.** SSE2 `F2`-prefixed `MOVSD` /
  `ADDSD` / `SUBSD` / `MULSD` / `DIVSD` plus `CVTSI2SD` and
  `CVTTSD2SI` for `i32 <-> f64` casts. 64-bit immediates land
  through a `[rsp-8]` scratch slot via two 32-bit stores. Tensor
  ops, packed f64 SIMD, and `print_f64` are still pending. ✓
- **`i8` and `u8` scalar codegen.** Held in 64-bit GPRs with a
  canonical low-byte representation: i8 sign-extends bits 8..63,
  u8 zero-extends them. Arithmetic re-canonicalises after each op
  with `MOVSX` / `MOVZX`; comparisons reuse the existing 64-bit
  signed/unsigned dispatch in `IsUnsigned()`. New cast ops cover
  every (i8/u8) × (i32/i64) direction. ✓
- **String literals** as a fat pointer `(ptr, len)`. CEIR keeps
  the value single-Value-shaped — a rodata pointer — and tracks
  the constant length in a side-table that the MIR lowering and
  the print_str call site consult to recover the second ABI
  register. `print_str("hello\n")` is a real compile-and-run. The
  multi-register call/return ABI for arbitrary string flow
  (concatenation, parameters, returns from non-print functions)
  is reserved for a later milestone with forward-pointing
  diagnostics. ✓

See [strings](strings).

## v0.13 — current

Effects and IO.

- **`@io` effect-typing.** Function signatures carry an effect set
  (`types.EffectSet`, currently a 1-bit bitmask for `@io`). The
  parser lexes attached `@<ident>` (no whitespace) as a dedicated
  attribute token, sidestepping the long-standing ambiguity with
  the matmul `@` operator. The type checker reads `@io` from the
  fn's `Attrs` and stores it on `FnSig`/`CheckedFn`. Call sites
  enforce `callee.Effects ⊆ caller.Effects`; `each` does the same
  on its second argument. ✓
- **`each` allowlist removed.** The hardcoded
  `print_i32`/`print_f32` set is gone. Any function whose effects
  fit the caller's effects is accepted, including pure user
  functions and user-defined `@io` wrappers around the print
  intrinsics. ✓
- Effect-polymorphic core combinators (`tabulate`, `scan`,
  `iterate_until`) — pending; their callbacks are pure today and
  the surface for declaring effect-polymorphic combinator inputs is
  reserved for a follow-up.

See [effects](effects).

## v0.14 — traits

- Element-type generics (`fn sum[T: Numeric, N](v: T[N]) -> T`).
- Trait registry, monomorphisation per `(shape, trait-impl)` pair.
- A small numeric stdlib: `Numeric`, `Float`, `Integer`, `Eq`, `Ord`.

See [traits](traits).

## v0.15 — linear types

- `mut`, `&`, `&mut`, `!` annotations.
- In-place ops (`+=`, `.+=`, `.*=`, `./=`).
- Dynamic-shape tensors with heap allocation, freed when the linear
  handle is consumed.
- `drop(x)` for explicit release.

See [linear types](linear-types).

## v0.16 — kernel DSL, GPU backend (initial)

- `kernel` declarations with `for` loops re-introduced inside them.
- Tile / block / per-axis loop combinators.
- Initial PTX backend: pure-tensor functions can be marked
  `on device` and compile to a single PTX kernel.
- `device` qualifier on tensor types.

See [kernel DSL](kernel-dsl) and [GPU backend](gpu-backend).

## v0.17+ — autodiff, full GPU surface

- Reverse-mode AD via `@grad` (single-parameter first, then
  multi-parameter / `@jvp`).
- VJP rules for the kernel DSL.
- Stream and event API for asynchronous GPU work.
- SPIR-V backend for portable GPU.

See [autodiff](autodiff).

## Cross-cutting work, no fixed version

These ride along when convenient:

- **Diagnostics depth**: more help text, span-correct multi-line
  notes, source-quote rendering.
- **Pattern exhaustiveness**: real coverage check (currently
  warning-only on a few patterns).
- **Broadcasting**: full NumPy-style rules with shape inference.
- **Stdlib**: the things you would expect in a numerics standard
  library — math (`sqrt`, `exp`, `log`), trig, linalg helpers.
- **REPL**: an interactive front-end that compiles each line to a
  small `.so` and `dlopen`s it.
