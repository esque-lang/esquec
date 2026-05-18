// Package ptx implements the NVIDIA PTX (Parallel Thread Execution) backend.
//
// PTX is a stable, portable intermediate representation for NVIDIA GPUs.
// It is compiled by the CUDA driver at runtime to target-specific GPU code.
//
// This package provides:
//   - PTX instruction definitions for compute operations
//   - Compiler from MIR to PTX assembly
//   - Module representation with kernels, functions, and globals
//   - PTX text emission for CUDA driver loading
//   - SSA-based register allocation (v0.9)
//   - Memory optimizations with shared memory and tiling (v0.9)
//
// # Architecture
//
// The PTX backend compiles MIR functions marked with @kernel to PTX kernels.
// Regular functions can also be compiled as device functions callable from kernels.
//
//	MIR Function (@kernel) → PTX Kernel (.entry)
//	MIR Function           → PTX Device Function (.func)
//
// # PTX Features
//
// Supported PTX features include:
//   - Register types: pred, b8, b16, b32, b64, f32, f64
//   - Memory spaces: global, shared, local, const, param
//   - Arithmetic: integer and floating-point operations
//   - Comparison and selection
//   - Memory access with state space annotations
//   - Control flow with predicated execution
//   - Synchronization barriers
//   - Device function calls
//   - Neural network operations (exp, log, relu, sigmoid, tanh)
//
// # Control Flow (v0.9)
//
// The PTX backend supports full control flow including loops and conditionals.
// Control flow is implemented using:
//   - Labels (OpLabel) for branch targets
//   - Predicated branches (OpBra with @predicate)
//   - Comparison instructions (OpSetp) to set predicates
//   - Select instructions (OpSelp) for conditional values
//
// Example control flow:
//
//	setp.lt.s32 p0, r1, r2;  // p0 = (r1 < r2)
//	@p0 bra loop_start;      // if p0 goto loop_start
//	bra loop_end;            // goto loop_end
//
// # Register Allocation (v0.9)
//
// The RegAlloc type provides SSA-based register allocation:
//
//	ra := NewRegAlloc()
//	alloc, err := ra.Allocate(mirFunc)
//	// alloc.GetReg(valueID, type) returns the assigned register
//
// The allocator:
//   - Computes liveness to determine when values die
//   - Reuses registers when values are no longer live
//   - Spills to local memory when registers are exhausted
//   - Supports up to 255 registers per type (PTX limit)
//
// # Memory Optimizations (v0.9)
//
// Shared memory and tiling optimizations for better performance:
//
//	mgr := NewSharedMemManager(DefaultMemoryConfig())
//	tileA, _ := mgr.Alloc("tileA", RegF32, 32*32)
//	insts, _ := TiledMatMul(mgr, "A", "B", "C", M, N, K)
//
// Memory features:
//   - Shared memory allocation with automatic offset management
//   - Tiled matrix multiplication using shared memory
//   - Coalesced memory access patterns
//   - Vector loads (ld.v2, ld.v4) for bandwidth optimization
//
// # Device Functions (v0.9)
//
// Compile device functions callable from kernels:
//
//	comp := NewCompiler()
//	fn, err := comp.CompileFunction(mirFunc)  // .func declaration
//	kernel, err := comp.CompileKernel(mirKernel)  // .entry declaration
//
// Device functions support:
//   - Parameters passed via .param space
//   - Return values via retval register
//   - Called using PTX call instruction
//
// # Example PTX Output
//
//	.version 8.0
//	.target sm_75
//	.address_size 64
//
//	.visible .entry vector_add(
//	    .param .u64 a,
//	    .param .u64 b,
//	    .param .u64 c
//	)
//	{
//	    .reg .b64 rd<3>;
//	    .reg .f32 f<3>;
//
//	    ld.param.u64 rd0, [a];
//	    ld.global.f32 f0, [rd0];
//	    ld.global.f32 f1, [rd1];
//	    add.f32 f2, f0, f1;
//	    st.global.f32 [rd2], f2;
//	    exit;
//	}
//
// # Integration
//
// PTX modules are loaded at runtime using the CUDA driver API via
// the cuda package. The workflow is:
//
//  1. Compile MIR to PTX using Compiler.CompileKernel()
//  2. Generate PTX text using Module.Emit()
//  3. Load PTX using cuda.LoadModule()
//  4. Get kernel function using Module.GetFunction()
//  5. Launch kernel using Function.Launch()
//
// # Neural Network Support
//
// Built-in support for neural network operations:
//   - Activation functions: relu, sigmoid, tanh
//   - Math functions: exp, log, sqrt, sin, cos
//   - All implemented using PTX special function units
//
// Example activation:
//
//	// relu(x) = max(0, x)
//	mov.f32 f1, 0.0;
//	max.f32 f2, f1, f0;  // f2 = relu(f0)
//
// Added in v0.8, enhanced in v0.9.
package ptx
