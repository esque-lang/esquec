---
title: CEIR
---

# CEIR — Core ESque IR

`internal/ceir`. Administrative-Normal-Form, single-static-assignment,
fully typed core IR. Every operand is a `*ceir.Value` (a name + a
type) or a literal.

## Shape

```
Module
 └── Functions
      └── Blocks
           └── Instructions   (one Op + operand Values)
                └── Result Value (with Type)
```

Every instruction has a single result (or none for `OpReturn`).
The IR is in ANF: nested expressions are flattened into named
intermediate `Value`s.

## Key operations

| Op             | Effect |
|----------------|--------|
| `OpConstInt`, `OpConstFloat`, `OpConstBool` | Materialise literals. |
| `OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpMod` | Scalar arithmetic. |
| `OpEAdd`, `OpEMul`, … | Element-wise tensor arithmetic. |
| `OpEq`, `OpNe`, `OpLt`, `OpLe`, `OpGt`, `OpGe` | Comparisons. |
| `OpAnd`, `OpOr`, `OpNot` | Logical. |
| `OpCall`        | Function call (dispatch on float-vs-int args). |
| `OpSelect`      | Ternary; powers `if`/`else` and the `iterate_until` cascade. |
| `OpTensorLit`   | Construct a tensor from N values. |
| `OpTensorElem`  | Read element `i` of a tensor by index. |
| `OpReduceAdd`   | Reduction (also `OpReduceMul`, etc). |
| `OpRodataPtr`   | Synthesised after const-folding tensor literals. |
| `OpCast`        | Numeric `as`. |
| `OpReturn`      | Function return. |

## Lowering

`internal/ceir/lower.go` translates AST → CEIR. Notable cases:

- **Tensor literal** → `OpTensorLit{Args: ...}`. Constant-only
  literals are spotted by the const-fold pass and replaced with
  `OpRodataPtr` pointing into a `RodataEntry`.
- **Range expression** `lo..hi` → an `OpTensorLit` of `OpConstInt`
  values for each element. Hits the rodata path automatically.
- **`tabulate(N, |i| f(i))`** → unroll the lambda body N times,
  each with `i` bound to a fresh `OpConstInt`. Collect into
  `OpTensorLit`. Pure-arithmetic lambdas constant-fold to rodata.
- **`scan(init, |a, x| f, v)`** → unroll, threading the
  accumulator across `OpTensorElem` calls and collecting all
  intermediate results.
- **`iterate_until(init, step, pred, max)`** → select-cascade
  unroll; see [Loop primitives → iterate_until](../language/loop-primitives#iterate_until).
- **`each(v, f)`** → unroll with `OpCall` per element; final
  result is unit.
- **Reductions `+/`** → `OpReduceAdd` (or sibling) on the operand.
  Element-wise pre-multiplication (`+/(a .* b)`) lowers to a
  fused pattern the backend recognises.

## Optimisation

`internal/ceir/opt.go` runs a constant-fold pass: every `OpAdd`
of two `OpConstInt` becomes a single `OpConstInt`, and so on for
all arithmetic and comparison ops. This pass is what makes
`tabulate(5, |i| i*i)` collapse to a single rodata literal.

## Inspecting

```bash
./esquec build foo.esq --emit=ceir
```

prints the IR in a human-readable form. Each instruction is one
line: `%dst = OpName %src1 %src2`.
