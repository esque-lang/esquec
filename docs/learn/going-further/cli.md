---
title: The esquec CLI
---

# The esquec CLI

`esquec` is the entire toolchain. It is a single static binary built
with `go build ./cmd/esquec`.

## Subcommands

| Command | Effect |
|---------|--------|
| `esquec build foo.esq -o foo`   | Compile and link to ELF executable. |
| `esquec check foo.esq`          | Type-check only, no codegen.       |

The flags listed below apply to `build`. `check` accepts only the
positional source file.

## Flags

| Flag                | Effect |
|---------------------|--------|
| `-o PATH`           | Output path for the executable or `.o` file. |
| `--emit=ast`        | Stop after parsing; print the AST. |
| `--emit=ceir`       | Stop after CEIR; print the core IR. |
| `--emit=mir`        | Stop after MIR. |
| `--emit=asm`        | Print hex of emitted machine code per fn. |
| `--emit=obj`        | Stop at the relocatable object; output is `.o`. |
| `--keep-obj`        | Keep the intermediate `.o` next to the executable. |

## Examples

Type-check without building:

```bash
./esquec check examples/02_functions.esq
```

Look at the AST:

```bash
./esquec build examples/02_functions.esq --emit=ast
```

Look at the core IR (this is the most informative dump for most
debugging):

```bash
./esquec build examples/02_functions.esq --emit=ceir
```

Look at machine code:

```bash
./esquec build examples/02_functions.esq --emit=asm
# Or, for a real disassembly:
./esquec build examples/02_functions.esq --emit=obj -o /tmp/sq.o
objdump -d /tmp/sq.o
```

## Project layout

esque does not yet have a notion of multi-file projects; you compile
one `.esq` at a time. Multi-module support is on the roadmap.

## Linking

`esquec` invokes the system `ld` with `-e _start` once it has
written the relocatable object. There is no runtime archive; the
`_start` stub is emitted directly into the user's `.o`. The only
external dependency at runtime is the kernel's syscall ABI.
