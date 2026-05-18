// Package runtime is the host-side runtime for compiled esque programs.
//
// In v0.0 the runtime is empty: the compiler synthesises a `_start`
// symbol directly that performs the `exit` syscall. As the language
// grows (heap, I/O, CUDA loader) the runtime gets actual code.
package runtime
