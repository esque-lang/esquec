---
title: Type checker
---

# Type checker

`internal/types`. Two-pass: first collect signatures, then check
bodies.

## Pass 1: signatures

`Check(*ast.File)` walks every top-level `*ast.FnDecl`, builds a
`*Function` for each, and registers it in the module's name table.
This pass:

- Resolves parameter and return types, including shape parameters.
- Validates shape-parameter declarations (no duplicates, correct
  kinds).
- Does **not** look at bodies. This is what makes mutual recursion
  and forward references trivial — every body sees every other
  signature.

## Pass 2: bodies

For each function, the checker walks the body bottom-up,
synthesising types and unifying against the declared return type.
The expression cases live in `checkExpr`. Each case:

- Synthesises types for sub-expressions.
- Validates the local rule (e.g. arithmetic operands must match;
  tensor element-wise operands must have equal shapes).
- Returns a `*types.Type` for the whole expression.

## Type representation

```go
type Type struct {
    K     Kind                // KI32, KF32, KBool, KUnit, KTensor
    Elem  *Type                // for KTensor: element type
    Shape []ShapeDim          // for KTensor: shape
}

type ShapeDim struct {
    Concrete int64    // -1 if not concrete
    VarName  string   // shape variable name; "" if concrete
}
```

`Type.Equal` compares structurally; `Type.WithShape(...)` lifts a
scalar to a tensor; `Type.Rank()` returns the number of dims.

## Monomorphisation

When a generic function `dot[N]` is called with a concrete shape
(e.g. `f32[3]`), the checker:

1. Resolves the binding `N=3`.
2. Looks up (or constructs) an `Instantiation` keyed by
   `dot__3`.
3. Type-checks the body with `N` substituted, producing a fresh
   `*Function` with `IsMonomorphized = true`,
   `OriginalName = "dot"`, `ShapeValues = {N:3}`.
4. The CEIR pass emits this function instead of (or alongside)
   the generic.

`InstKey` produces the deterministic name (`dot__3`,
`matmul__3_4_2`, …). Keys are sorted by shape-parameter name
alphabetically.

## Integer suffix dispatch

`checkIntLit` handles `_i32`, `_i64`, `_u32`, `_i8`, and `_u8`
literals as of v0.12. The remaining suffixes (`_i16`, `_u16`,
`_u64`) are parsed and range-checked but produce a forward-pointing
diagnostic in the type checker — `intSuffixLimits` knows their
valid ranges, but `supportedIntSuffixes` is the gating set the
checker compares against. Adding a new width to the codegen means:
add an entry to `supportedIntSuffixes`, add a `Type` factory in
`types.go` if it doesn't exist, route `IntLit.TypeSuffix` to it
from `ceir.lower.go`, and confirm the backend has the relevant 8 /
16 / 32 / 64-bit encoder forms.

`*Type.IsUnsigned()` is the canonical query the backend uses to
choose between `IDIV`/`DIV` and signed/unsigned condition codes.

## Effects (`@io`, v0.13)

Function signatures carry an `EffectSet`, a small bitmask declared in
`internal/types/types.go`. Today the only bit is `EffIO`; `@alloc` /
`@panic` / `@nondet` ride along the same representation in later
milestones. The empty set means "pure".

The pipeline is:

1. **Lex**. `@<ident>` (no whitespace between `@` and the ident) lexes
   as a single `TkAttr` token whose `Lit` is the attribute name. Bare
   `@` (with surrounding whitespace) stays as `TkAt` and remains the
   matmul operator. This is what makes `print_i32(1)` and a following
   `@io fn` line unambiguous.
2. **Parse**. `parseAttrs` reads `TkAttr` (and the legacy `TkKernel`
   fast-path token) into `ast.Attr` nodes attached to the `FnDecl`.
3. **Type-check pass 1**. `effectsFromAttrs` distils each `FnDecl`'s
   `Attrs` into an `EffectSet`. The result is stored on the `FnSig`
   along with the parameter and return types. Runtime intrinsics
   (`print_i32`, `print_f32`, `print_str`) are seeded with `EffIO` in
   `registerRuntimeBuiltins`.
4. **Type-check pass 2**. Every `*ast.Call` and every `*ast.Each`
   verifies `callee.Effects.Subset(caller.Effects)`. The caller's
   effect set is threaded through `checkCtx.fnEffects`. A pure caller
   that touches an `@io` callee gets a diagnostic that names both
   sides and suggests adding the annotation.
5. **Lowering and codegen**. CEIR / MIR / x86 don't see effects: the
   type checker is the only enforcement point. By the time CEIR
   lowers `each(v, fn)`, `fn` is just a name in `OpEachLoop`'s
   `FnName` field — the backend dispatches on relocations and
   `IsPrintBuiltin` to decide whether to synthesize a body.

## Diagnostics

Type errors are `*diag.Diagnostic` instances with the offending
expression's span. Many include forward-pointing help lines
(e.g. "_u16 is parseable but not yet supported by the v0.12 CPU
backend; only _i8, _u8, _i32, _i64, and _u32 currently codegen").
