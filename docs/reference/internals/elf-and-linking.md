---
title: ELF and linking
---

# ELF and linking

The compiler produces a relocatable ELF64 `.o` file in pure Go, then
shells out to the system `ld` for the final link.

## Object writer

`internal/object/elf/elf.go`. Produces relocatable objects with the
following sections, in order:

| Section          | Contents                                          |
|------------------|---------------------------------------------------|
| `.text`          | Code; one symbol per fn (and `_start`).           |
| `.rodata`        | Constant tensor data (only when present).         |
| `.note.GNU-stack`| Empty; marks the stack non-executable.            |
| `.symtab`        | Symbol table.                                     |
| `.strtab`        | String table for symbol names.                    |
| `.rela.text`     | Relocations against `.text`.                      |

Relocation types emitted:

- `R_X86_64_PLT32` — PC-relative call site.
- `R_X86_64_PC32`  — PC-relative load (used for rodata `lea`).

## Symbols

Each user-defined function becomes a `STT_FUNC` symbol with the
fn's name. Each rodata blob becomes `STT_OBJECT`. The driver also
emits a separate `_start` global symbol pointing at the entry
function so that `ld -e _start` resolves cleanly.

`_start` itself is emitted directly into `.text` by the backend; it
calls `main`, then issues a Linux `exit` syscall (number 60) with
`main`'s return value as the exit status. There is no libc.

## Linker driver

`internal/link/link.go`. A thin wrapper that builds the `ld`
argument list, invokes it, and reports its exit code.

```
ld -o <out> -e _start <obj> [<runtime objects>]
```

There are no separate runtime objects to link in beyond what the
driver appends to the user's `.o` directly (the `print_*`
intrinsics — see [x86 backend → Runtime intrinsics](x86)). There
is no link-time optimisation step.

## Inspection

```
./esquec build foo.esq --emit=obj -o foo.o
objdump -h foo.o            # section headers
objdump -t foo.o            # symbol table
objdump -d foo.o            # disassembly
objdump -r foo.o            # relocations
objdump -s -j .rodata foo.o # raw rodata bytes
```

The `--keep-obj` flag preserves the `.o` next to the executable.
