// Package cuda provides CUDA runtime integration for GPU kernel execution.
//
// This package interfaces with the NVIDIA CUDA driver API (libcuda.so) to:
//   - Initialize CUDA devices
//   - Load PTX modules compiled from esque code
//   - Launch GPU kernels
//   - Manage device memory
//
// # Overview
//
// The CUDA package uses dynamic linking (dlopen) to load the CUDA driver
// at runtime, avoiding compile-time dependencies on the CUDA toolkit.
// This allows esquec binaries to run on systems without CUDA installed
// (with GPU features disabled).
//
// # Basic Usage
//
//	// Initialize CUDA
//	if err := cuda.Init(); err != nil {
//	    log.Fatal("CUDA not available:", err)
//	}
//
//	// Get device
//	dev, _ := cuda.GetDevice(0)
//	ctx, _ := dev.CreateContext()
//	defer ctx.Destroy()
//
//	// Load PTX module
//	mod, _ := cuda.LoadModule(ptxSource)
//	defer mod.Unload()
//
//	// Get kernel function
//	fn, _ := mod.GetFunction("vector_add")
//
//	// Allocate device memory
//	devA, _ := cuda.Alloc(1024)
//	defer devA.Free()
//
//	// Copy data to device
//	devA.CopyFromHost(hostData)
//
//	// Launch kernel
//	params := []unsafe.Pointer{
//	    unsafe.Pointer(&devA.Ptr()),
//	}
//	fn.Launch(
//	    numBlocks, 1, 1,    // grid dimensions
//	    threadsPerBlock, 1, 1,  // block dimensions
//	    0,                  // shared memory
//	    params,
//	)
//
//	// Synchronize and copy results
//	cuda.Synchronize()
//	devA.CopyToHost(results)
//
// # Device Management
//
// Query available devices:
//
//	count, _ := cuda.DeviceCount()
//	for i := 0; i < count; i++ {
//	    dev, _ := cuda.GetDevice(i)
//	    name, _ := dev.Name()
//	    fmt.Printf("Device %d: %s\n", i, name)
//	}
//
// # Memory Management
//
// Device memory is allocated and managed explicitly:
//
//	// Allocate 1MB
//	ptr, err := cuda.Alloc(1024 * 1024)
//	if err != nil {
//	    // Handle out-of-memory
//	}
//	defer ptr.Free()
//
//	// Copy to device
//	ptr.CopyFromHost(data)
//
//	// Copy from device
//	ptr.CopyToHost(result)
//
// # Error Handling (v0.9)
//
// The package provides comprehensive error handling:
//
//	err := fn.Launch(...)
//	if cuda.IsMemoryError(err) {
//	    // Handle out of memory
//	}
//	if cuda.IsKernelError(err) {
//	    // Handle kernel execution failure
//	}
//
//	// Get suggestions for fixing errors
//	if suggestions := cuda.Suggestions(err); len(suggestions) > 0 {
//	    for _, s := range suggestions {
//	        fmt.Println("Try:", s)
//	    }
//	}
//
// Error categories:
//   - Memory errors: out of memory, illegal address
//   - Kernel errors: launch failure, timeout, resource exhaustion
//   - PTX errors: invalid PTX, compilation failure
//   - Device errors: no device, invalid device
//   - Driver errors: library not found, symbol not found
//
// # Kernel Launch Configuration
//
// Kernels are launched with grid and block dimensions:
//
//	// 1D launch: 256 threads total, 64 per block
//	fn.Launch(4, 1, 1, 64, 1, 1, 0, params)
//
//	// 2D launch: 16x16 blocks of 16x16 threads
//	fn.Launch(16, 16, 1, 16, 16, 1, 0, params)
//
//	// With shared memory: 48KB per block
//	fn.Launch(numBlocks, 1, 1, 256, 1, 1, 48*1024, params)
//
// # Thread Safety
//
// CUDA contexts are not thread-safe. Each OS thread should have its
// own context, or contexts should be made current explicitly:
//
//	ctx.SetCurrent()  // Make context current for this thread
//
// # Availability Check
//
// Check if CUDA is available before using:
//
//	if cuda.Available() {
//	    // Use GPU acceleration
//	} else {
//	    // Fall back to CPU
//	}
//
// # Requirements
//
// Runtime requirements:
//   - NVIDIA GPU (compute capability 3.0+)
//   - NVIDIA driver with libcuda.so
//   - Linux (Windows support planned)
//
// Compile requirements:
//   - CGO enabled
//   - libdl for dynamic loading
//
// Added in v0.8, enhanced in v0.9.
package cuda
