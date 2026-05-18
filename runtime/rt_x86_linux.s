// Reserved for v0.1+: a real `_start` that calls `main`, sets up
// argc/argv/envp, and exits with main's return value. v0.0 emits its
// own `_start` directly from the backend, so this file is currently
// informational.
//
// .text
// .global _start
// _start:
//     # rsp -> argc
//     # call main
//     # mov %eax, %edi
//     # mov $60, %eax    # sys_exit
//     # syscall
