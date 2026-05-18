// Package cuda provides CUDA runtime integration via cgo.
// This package enables loading PTX modules and launching kernels.
//
// The CUDA driver API is used (libcuda.so) rather than the runtime API
// to avoid requiring the full CUDA toolkit at compile time.
package cuda

/*
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

// CUDA types (from cuda.h)
typedef int CUresult;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUdeviceptr;
typedef void* CUstream;
typedef int CUdevice;

// CUDA error codes
#define CUDA_SUCCESS 0

// Function pointer types
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void**, void**);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuGetErrorString_t)(CUresult, const char**);

// Global function pointers (loaded at runtime)
static void* cuda_lib = NULL;
static cuInit_t pfn_cuInit = NULL;
static cuDeviceGet_t pfn_cuDeviceGet = NULL;
static cuDeviceGetCount_t pfn_cuDeviceGetCount = NULL;
static cuDeviceGetName_t pfn_cuDeviceGetName = NULL;
static cuCtxCreate_t pfn_cuCtxCreate = NULL;
static cuCtxDestroy_t pfn_cuCtxDestroy = NULL;
static cuCtxSetCurrent_t pfn_cuCtxSetCurrent = NULL;
static cuModuleLoadData_t pfn_cuModuleLoadData = NULL;
static cuModuleUnload_t pfn_cuModuleUnload = NULL;
static cuModuleGetFunction_t pfn_cuModuleGetFunction = NULL;
static cuMemAlloc_t pfn_cuMemAlloc = NULL;
static cuMemFree_t pfn_cuMemFree = NULL;
static cuMemcpyHtoD_t pfn_cuMemcpyHtoD = NULL;
static cuMemcpyDtoH_t pfn_cuMemcpyDtoH = NULL;
static cuLaunchKernel_t pfn_cuLaunchKernel = NULL;
static cuCtxSynchronize_t pfn_cuCtxSynchronize = NULL;
static cuGetErrorString_t pfn_cuGetErrorString = NULL;

// Load CUDA driver library
static int cuda_load_library() {
    if (cuda_lib != NULL) return 0;

    cuda_lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!cuda_lib) cuda_lib = dlopen("libcuda.so", RTLD_NOW);
    if (!cuda_lib) return -1;

    pfn_cuInit = (cuInit_t)dlsym(cuda_lib, "cuInit");
    pfn_cuDeviceGet = (cuDeviceGet_t)dlsym(cuda_lib, "cuDeviceGet");
    pfn_cuDeviceGetCount = (cuDeviceGetCount_t)dlsym(cuda_lib, "cuDeviceGetCount");
    pfn_cuDeviceGetName = (cuDeviceGetName_t)dlsym(cuda_lib, "cuDeviceGetName");
    pfn_cuCtxCreate = (cuCtxCreate_t)dlsym(cuda_lib, "cuCtxCreate_v2");
    pfn_cuCtxDestroy = (cuCtxDestroy_t)dlsym(cuda_lib, "cuCtxDestroy_v2");
    pfn_cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda_lib, "cuCtxSetCurrent");
    pfn_cuModuleLoadData = (cuModuleLoadData_t)dlsym(cuda_lib, "cuModuleLoadData");
    pfn_cuModuleUnload = (cuModuleUnload_t)dlsym(cuda_lib, "cuModuleUnload");
    pfn_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(cuda_lib, "cuModuleGetFunction");
    pfn_cuMemAlloc = (cuMemAlloc_t)dlsym(cuda_lib, "cuMemAlloc_v2");
    pfn_cuMemFree = (cuMemFree_t)dlsym(cuda_lib, "cuMemFree_v2");
    pfn_cuMemcpyHtoD = (cuMemcpyHtoD_t)dlsym(cuda_lib, "cuMemcpyHtoD_v2");
    pfn_cuMemcpyDtoH = (cuMemcpyDtoH_t)dlsym(cuda_lib, "cuMemcpyDtoH_v2");
    pfn_cuLaunchKernel = (cuLaunchKernel_t)dlsym(cuda_lib, "cuLaunchKernel");
    pfn_cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda_lib, "cuCtxSynchronize");
    pfn_cuGetErrorString = (cuGetErrorString_t)dlsym(cuda_lib, "cuGetErrorString");

    if (!pfn_cuInit || !pfn_cuDeviceGet || !pfn_cuCtxCreate ||
        !pfn_cuModuleLoadData || !pfn_cuModuleGetFunction ||
        !pfn_cuMemAlloc || !pfn_cuMemFree || !pfn_cuLaunchKernel) {
        dlclose(cuda_lib);
        cuda_lib = NULL;
        return -2;
    }

    return 0;
}

// Wrapper functions
static int cuda_init() {
    if (cuda_load_library() != 0) return -1;
    return pfn_cuInit(0);
}

static int cuda_device_count(int* count) {
    if (!pfn_cuDeviceGetCount) return -1;
    return pfn_cuDeviceGetCount(count);
}

static int cuda_device_get(CUdevice* dev, int ordinal) {
    if (!pfn_cuDeviceGet) return -1;
    return pfn_cuDeviceGet(dev, ordinal);
}

static int cuda_device_name(char* name, int len, CUdevice dev) {
    if (!pfn_cuDeviceGetName) return -1;
    return pfn_cuDeviceGetName(name, len, dev);
}

static int cuda_ctx_create(CUcontext* ctx, CUdevice dev) {
    if (!pfn_cuCtxCreate) return -1;
    return pfn_cuCtxCreate(ctx, 0, dev);
}

static int cuda_ctx_destroy(CUcontext ctx) {
    if (!pfn_cuCtxDestroy) return -1;
    return pfn_cuCtxDestroy(ctx);
}

static int cuda_ctx_set_current(CUcontext ctx) {
    if (!pfn_cuCtxSetCurrent) return -1;
    return pfn_cuCtxSetCurrent(ctx);
}

static int cuda_module_load(CUmodule* mod, const char* ptx) {
    if (!pfn_cuModuleLoadData) return -1;
    return pfn_cuModuleLoadData(mod, ptx);
}

static int cuda_module_unload(CUmodule mod) {
    if (!pfn_cuModuleUnload) return -1;
    return pfn_cuModuleUnload(mod);
}

static int cuda_get_function(CUfunction* func, CUmodule mod, const char* name) {
    if (!pfn_cuModuleGetFunction) return -1;
    return pfn_cuModuleGetFunction(func, mod, name);
}

static int cuda_mem_alloc(CUdeviceptr* ptr, size_t size) {
    if (!pfn_cuMemAlloc) return -1;
    return pfn_cuMemAlloc(ptr, size);
}

static int cuda_mem_free(CUdeviceptr ptr) {
    if (!pfn_cuMemFree) return -1;
    return pfn_cuMemFree(ptr);
}

static int cuda_memcpy_htod(CUdeviceptr dst, const void* src, size_t size) {
    if (!pfn_cuMemcpyHtoD) return -1;
    return pfn_cuMemcpyHtoD(dst, src, size);
}

static int cuda_memcpy_dtoh(void* dst, CUdeviceptr src, size_t size) {
    if (!pfn_cuMemcpyDtoH) return -1;
    return pfn_cuMemcpyDtoH(dst, src, size);
}

static int cuda_launch(CUfunction func,
                       unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                       unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                       unsigned int sharedMem, void** params) {
    if (!pfn_cuLaunchKernel) return -1;
    return pfn_cuLaunchKernel(func, gridX, gridY, gridZ,
                              blockX, blockY, blockZ,
                              sharedMem, NULL, params, NULL);
}

static int cuda_synchronize() {
    if (!pfn_cuCtxSynchronize) return -1;
    return pfn_cuCtxSynchronize();
}

static const char* cuda_error_string(int err) {
    const char* str = NULL;
    if (pfn_cuGetErrorString && pfn_cuGetErrorString(err, &str) == 0) {
        return str;
    }
    return "unknown error";
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// cudaError converts a C error code to a Go error with context.
func cudaError(code C.int) error {
	return Check(int(code), "CUDA operation")
}

// cudaErrorCtx converts a C error code to a Go error with specific context.
func cudaErrorCtx(code C.int, context string) error {
	return Check(int(code), context)
}

// Init initializes the CUDA driver.
func Init() error {
	return cudaErrorCtx(C.cuda_init(), "initializing CUDA driver")
}

// DeviceCount returns the number of CUDA devices.
func DeviceCount() (int, error) {
	var count C.int
	if err := cudaErrorCtx(C.cuda_device_count(&count), "querying device count"); err != nil {
		return 0, err
	}
	return int(count), nil
}

// Device represents a CUDA device.
type Device struct {
	handle C.CUdevice
}

// GetDevice returns the device at the given ordinal.
func GetDevice(ordinal int) (*Device, error) {
	var dev C.CUdevice
	if err := cudaError(C.cuda_device_get(&dev, C.int(ordinal))); err != nil {
		return nil, err
	}
	return &Device{handle: dev}, nil
}

// Name returns the device name.
func (d *Device) Name() (string, error) {
	name := make([]byte, 256)
	if err := cudaError(C.cuda_device_name((*C.char)(unsafe.Pointer(&name[0])), 256, d.handle)); err != nil {
		return "", err
	}
	// Find null terminator
	for i, b := range name {
		if b == 0 {
			return string(name[:i]), nil
		}
	}
	return string(name), nil
}

// Context represents a CUDA context.
type Context struct {
	handle C.CUcontext
}

// CreateContext creates a new CUDA context on the device.
func (d *Device) CreateContext() (*Context, error) {
	var ctx C.CUcontext
	if err := cudaError(C.cuda_ctx_create(&ctx, d.handle)); err != nil {
		return nil, err
	}
	return &Context{handle: ctx}, nil
}

// Destroy destroys the context.
func (c *Context) Destroy() error {
	return cudaError(C.cuda_ctx_destroy(c.handle))
}

// SetCurrent makes this context current.
func (c *Context) SetCurrent() error {
	return cudaError(C.cuda_ctx_set_current(c.handle))
}

// Module represents a loaded PTX module.
type Module struct {
	handle C.CUmodule
}

// LoadModule loads a PTX module from source.
func LoadModule(ptx string) (*Module, error) {
	cptx := C.CString(ptx)
	defer C.free(unsafe.Pointer(cptx))

	var mod C.CUmodule
	if err := cudaErrorCtx(C.cuda_module_load(&mod, cptx), "loading PTX module"); err != nil {
		return nil, err
	}
	return &Module{handle: mod}, nil
}

// Unload unloads the module.
func (m *Module) Unload() error {
	return cudaError(C.cuda_module_unload(m.handle))
}

// GetFunction gets a kernel function from the module.
func (m *Module) GetFunction(name string) (*Function, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	var fn C.CUfunction
	if err := cudaError(C.cuda_get_function(&fn, m.handle, cname)); err != nil {
		return nil, err
	}
	return &Function{handle: fn}, nil
}

// Function represents a CUDA kernel function.
type Function struct {
	handle C.CUfunction
}

// Launch launches the kernel with the given grid and block dimensions.
func (f *Function) Launch(gridX, gridY, gridZ, blockX, blockY, blockZ int, sharedMem int, params []unsafe.Pointer) error {
	var cparams *unsafe.Pointer
	if len(params) > 0 {
		cparams = (*unsafe.Pointer)(unsafe.Pointer(&params[0]))
	}
	return cudaErrorCtx(C.cuda_launch(f.handle,
		C.uint(gridX), C.uint(gridY), C.uint(gridZ),
		C.uint(blockX), C.uint(blockY), C.uint(blockZ),
		C.uint(sharedMem), cparams),
		fmt.Sprintf("launching kernel with grid(%d,%d,%d) block(%d,%d,%d)",
			gridX, gridY, gridZ, blockX, blockY, blockZ))
}

// DevicePtr represents a pointer to device memory.
type DevicePtr struct {
	ptr  C.CUdeviceptr
	size int64
}

// Alloc allocates device memory.
func Alloc(size int64) (*DevicePtr, error) {
	var ptr C.CUdeviceptr
	if err := cudaErrorCtx(C.cuda_mem_alloc(&ptr, C.size_t(size)),
		fmt.Sprintf("allocating %d bytes of device memory", size)); err != nil {
		return nil, err
	}
	return &DevicePtr{ptr: ptr, size: size}, nil
}

// Free frees device memory.
func (d *DevicePtr) Free() error {
	return cudaError(C.cuda_mem_free(d.ptr))
}

// CopyFromHost copies data from host to device.
func (d *DevicePtr) CopyFromHost(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return cudaError(C.cuda_memcpy_htod(d.ptr, unsafe.Pointer(&data[0]), C.size_t(len(data))))
}

// CopyToHost copies data from device to host.
func (d *DevicePtr) CopyToHost(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return cudaError(C.cuda_memcpy_dtoh(unsafe.Pointer(&data[0]), d.ptr, C.size_t(len(data))))
}

// Ptr returns the raw device pointer as uintptr.
func (d *DevicePtr) Ptr() uintptr {
	return uintptr(d.ptr)
}

// Synchronize waits for all operations to complete.
func Synchronize() error {
	return cudaError(C.cuda_synchronize())
}

// Available reports whether CUDA is available on this system.
func Available() bool {
	if err := Init(); err != nil {
		return false
	}
	count, err := DeviceCount()
	if err != nil {
		return false
	}
	return count > 0
}
