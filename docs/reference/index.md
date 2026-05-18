---
title: Reference
slug: /
---

# Reference

The reference is for readers who already know the basics — if you
have not yet, the [Learn track](/) is the front door. Reference pages
are short, dense, and assume the rest of the language is already
familiar.

The reference has four sections.

## [Language reference](language/lexical)

The grammar, types, operators, and built-in primitives, exactly as
the implementation accepts them.

- [Lexical structure](language/lexical)
- [Operators and precedence](language/operators)
- [Type system](language/types)
- [Functions](language/functions)
- [Tensors](language/tensors)
- [Loop primitives](language/loop-primitives)
- [Pattern matching](language/pattern-matching)
- [Intrinsics](language/intrinsics)
- [Grammar (EBNF)](language/grammar)

## [Compiler internals](internals/pipeline)

The shape of the implementation: the stages, what each one does, and
where you would look in the source tree.

- [Pipeline overview](internals/pipeline)
- [Lexer](internals/lexer) · [Parser](internals/parser) · [Type checker](internals/types)
- [CEIR](internals/ceir) · [MIR](internals/mir) · [x86 backend](internals/x86)
- [ELF and linking](internals/elf-and-linking)
- [Diagnostics](internals/diagnostics)

## [Specification](spec/overview)

The semi-formal description of what the language *means*.
Implementation-independent in intent; written in service of a future
second implementation.

- [Overview](spec/overview)
- [Lexical](spec/lexical) · [Types](spec/types) · [Expressions](spec/expressions)
- [Functions](spec/functions) · [Tensors](spec/tensors) · [Modules](spec/modules)

## [Planned features](planned/overview)

Features in the spec or the roadmap that are not yet fully
implemented. Each page describes the design intent, what (if
anything) has already shipped, and where it sits on the
[roadmap](planned/roadmap). A few of these pages — `@io` effects
and string literals — describe machinery that has *already* landed
in part; they remain here because their full surface is still
ahead.

- [Effect system (`@io`)](planned/effects)
- [Strings and bytes](planned/strings)
- [Extended numerics (i64, u8, …)](planned/extended-numerics)
- [Traits](planned/traits)
- [Linear types](planned/linear-types)
- [Kernel DSL](planned/kernel-dsl)
- [GPU backend](planned/gpu-backend)
- [Autodiff](planned/autodiff)
- [Roadmap](planned/roadmap)
