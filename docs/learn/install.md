---
title: Install
---

# Install

esque currently targets **Linux x86-64 only**. The compiler shells out to
the system `ld` to produce final executables; everything before that is
self-contained.

## Prerequisites

- Go 1.25 or later (`go version`)
- A working `ld` — present on essentially every Linux system that has
  glibc or musl
- A CPU with SSE2 (every x86-64 CPU). AVX2 is detected and used when
  available; without it the SSE path is taken automatically

If you want to run the benchmarks against C, you also want `gcc` with
`-mavx2` support. That is optional.

## Get the source

```bash
git clone https://github.com/esque-lang/esquec.git
cd esquec
```

## Build the compiler

```bash
go build -o esquec ./cmd/esquec
```

That single binary is the whole toolchain. Drop it on your `$PATH` or
keep it in the repo.

## Sanity-check

```bash
./esquec build examples/02_functions.esq -o /tmp/sq
/tmp/sq; echo $?
# 25
```

The example computes `3² + 4²` and exits with the result. If you saw
`25`, you are ready.

## Run the test suite (optional)

```bash
go test ./...
```

This runs unit tests for every package plus the end-to-end suite in
`tests/e2e/`, which compiles small `.esq` programs, runs them, and
asserts on exit codes (or stdout, for `print_*` intrinsics).

## What you do **not** need

- LLVM
- A C toolchain (other than the linker itself)
- A package manager beyond `go`
- Any runtime library to link your programs against

The only external program esque invokes at compile time is `ld`.

## Next: [Hello, world](hello-world)
