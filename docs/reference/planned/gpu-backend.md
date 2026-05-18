---
title: GPU backend
---

# GPU backend

**Target version:** v0.16+.

## What

A second backend, after x86-64 CPU, that compiles esque to GPU
machine code. The current candidate is **PTX** (NVIDIA), with
SPIR-V (Vulkan / OpenCL) as a future target.

The CEIR is, by design, mostly device-agnostic. The GPU backend
takes the same CEIR the CPU backend takes, lowers it to a
GPU-flavoured MIR, and emits PTX text that NVIDIA's driver
JITs to SASS at load time.

## Status

`internal/backend/gpu/{ptx,cuda}` exists today as scaffolding —
data structures and naming, no actual codegen. It is *not* on the
default build path; `go build ./cmd/esquec` does not exercise it.

## Why not today

Three reasons:

1. **Kernel DSL first.** Writing performance-portable GPU code
   requires the kernel DSL surface, which is itself
   [planned](kernel-dsl).
2. **Device qualifiers in the type system.** Tensors need to
   carry `on host` / `on device` annotations, and the planner
   has to insert transfers. That is a meaningful type-system
   addition.
3. **Stream and event API.** Real GPU work needs asynchronous
   queues, events for ordering, and pinned host memory for DMA.

## Slice that may ship earlier

A simpler path is "compile-pure-tensor-code-to-GPU-as-one-kernel":
take a function whose body is purely tensor expressions (no
recursion, no `iterate_until`), compile the whole body to a single
PTX kernel, launch it from a thin host stub. That is much smaller
than the full kernel DSL story and may land first, before the
"write your own kernel" surface.

## Long term

Beyond NVIDIA: SPIR-V (Vulkan / OpenCL), Metal (Apple), or wgpu.
Each is roughly the same shape — a target IR, a launch ABI, a
device-side memory model. The CEIR-as-source design keeps that
work additive.
