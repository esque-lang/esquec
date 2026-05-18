---
title: Modules
---

# Spec: modules

## File model

A *module* today is one source file. The file consists of zero or
more top-level function declarations:

```
File = { FnDecl }
```

There are no imports today. There is no notion of a package or a
multi-file unit.

## Entry point

The entry point is a function named `main` with no parameters and a
return type of `i32`. The implementation is required to invoke
`main` at program start and use its return value as the process
exit code.

## Visibility

All top-level declarations in a file are visible to every other
top-level declaration in the same file. There is no `pub` /
private distinction today.

## Multi-file projects **(planned)**

The plan is:

- A *module* will be a single file, declared with a top-of-file
  `module name`.
- A *package* will be a directory of modules, with module-level
  `import` statements.
- Visibility will be controlled with `pub`.

The full design is **(planned)**; see
[planned: roadmap](/reference/planned/roadmap).

## Constants **(planned)**

Top-level `const` declarations of kind `nat` will be usable in
shape positions. Today, shapes must be either literals, shape
parameters, or simple shape arithmetic; module-level constants of
nat-kind are not yet supported.

## Attributes

The function-attribute syntax `@<ident>` shipped in v0.13. The only
attribute the type checker enforces today is `@io`, the I/O effect
(see [Functions: purity and effects](functions) and
[Effects (`@io`)](/reference/planned/effects)). Other attributes
(`@inline`, `@noinline`, `@kernel`, …) are reserved by the
grammar but currently ignored by later passes — they are
**(planned)**.
