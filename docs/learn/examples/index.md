---
title: Worked examples
---

# Worked examples

Each example below is a complete, runnable `.esq` program. Save it,
compile it with `./esquec build foo.esq -o foo`, run it. Each page
explains what the program does, the language ideas it exercises, and
how to read the generated code if you are curious.

The examples are roughly ordered by complexity:

| #  | Page | Idea |
|----|------|------|
| 1 | [Exit status](exit-status) | shortest possible programs; arithmetic |
| 2 | [Recursion](recursion) | self-recursive functions, conditionals |
| 3 | [Dot product](dot-product) | shape-generic tensor function |
| 4 | [Euclidean distance](euclidean-distance) | composing tensor ops |
| 5 | [Scan / prefix sum](scan-prefix-sum) | running accumulators |
| 6 | [Iterate until](iterate-until) | bounded fixpoint |

Every fixture also lives under `tests/e2e/` in the repo, where it has
an associated test that asserts on the exit code and (where relevant)
stdout.
