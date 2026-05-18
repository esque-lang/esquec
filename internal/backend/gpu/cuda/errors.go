// Package cuda provides comprehensive CUDA error handling.
package cuda

import (
	"fmt"
)

// ErrorCode represents CUDA error codes.
type ErrorCode int

// CUDA error codes (from cuda.h)
const (
	Success                      ErrorCode = 0
	ErrInvalidValue              ErrorCode = 1
	ErrOutOfMemory               ErrorCode = 2
	ErrNotInitialized            ErrorCode = 3
	ErrDeinitialized             ErrorCode = 4
	ErrProfilerDisabled          ErrorCode = 5
	ErrNoDevice                  ErrorCode = 100
	ErrInvalidDevice             ErrorCode = 101
	ErrDeviceNotLicensed         ErrorCode = 102
	ErrInvalidImage              ErrorCode = 200
	ErrInvalidContext            ErrorCode = 201
	ErrContextAlreadyCurrent     ErrorCode = 202
	ErrMapFailed                 ErrorCode = 205
	ErrUnmapFailed               ErrorCode = 206
	ErrArrayIsMapped             ErrorCode = 207
	ErrAlreadyMapped             ErrorCode = 208
	ErrNoBinaryForGpu            ErrorCode = 209
	ErrAlreadyAcquired           ErrorCode = 210
	ErrNotMapped                 ErrorCode = 211
	ErrNotMappedAsArray          ErrorCode = 212
	ErrNotMappedAsPointer        ErrorCode = 213
	ErrECCUncorrectable          ErrorCode = 214
	ErrUnsupportedLimit          ErrorCode = 215
	ErrContextAlreadyInUse       ErrorCode = 216
	ErrPeerAccessUnsupported     ErrorCode = 217
	ErrInvalidPTX                ErrorCode = 218
	ErrInvalidGraphicsContext    ErrorCode = 219
	ErrNvlinkUncorrectable       ErrorCode = 220
	ErrJitCompilerNotFound       ErrorCode = 221
	ErrInvalidSource             ErrorCode = 300
	ErrFileNotFound              ErrorCode = 301
	ErrSharedObjectSymbolNotFound ErrorCode = 302
	ErrSharedObjectInitFailed    ErrorCode = 303
	ErrOperatingSystem           ErrorCode = 304
	ErrInvalidHandle             ErrorCode = 400
	ErrIllegalState              ErrorCode = 401
	ErrNotFound                  ErrorCode = 500
	ErrNotReady                  ErrorCode = 600
	ErrIllegalAddress            ErrorCode = 700
	ErrLaunchOutOfResources      ErrorCode = 701
	ErrLaunchTimeout             ErrorCode = 702
	ErrLaunchIncompatibleTexturing ErrorCode = 703
	ErrPeerAccessAlreadyEnabled  ErrorCode = 704
	ErrPeerAccessNotEnabled      ErrorCode = 705
	ErrPrimaryContextActive      ErrorCode = 708
	ErrContextIsDestroyed        ErrorCode = 709
	ErrAssert                    ErrorCode = 710
	ErrTooManyPeers              ErrorCode = 711
	ErrHostMemoryAlreadyRegistered ErrorCode = 712
	ErrHostMemoryNotRegistered   ErrorCode = 713
	ErrHardwareStackError        ErrorCode = 714
	ErrIllegalInstruction        ErrorCode = 715
	ErrMisalignedAddress         ErrorCode = 716
	ErrInvalidAddressSpace       ErrorCode = 717
	ErrInvalidPC                 ErrorCode = 718
	ErrLaunchFailed              ErrorCode = 719
	ErrCooperativeLaunchTooLarge ErrorCode = 720
	ErrNotPermitted              ErrorCode = 800
	ErrNotSupported              ErrorCode = 801
	ErrSystemNotReady            ErrorCode = 802
	ErrSystemDriverMismatch      ErrorCode = 803
	ErrCompatNotSupportedOnDevice ErrorCode = 804
	ErrStreamCaptureUnsupported  ErrorCode = 900
	ErrStreamCaptureInvalidated  ErrorCode = 901
	ErrStreamCaptureMerge        ErrorCode = 902
	ErrStreamCaptureUnmatched    ErrorCode = 903
	ErrStreamCaptureUnjoined     ErrorCode = 904
	ErrStreamCaptureIsolation    ErrorCode = 905
	ErrStreamCaptureImplicit     ErrorCode = 906
	ErrCapturedEvent             ErrorCode = 907
	ErrStreamCaptureWrongThread  ErrorCode = 908
	ErrTimeout                   ErrorCode = 909
	ErrGraphExecUpdateFailure    ErrorCode = 910
	ErrUnknown                   ErrorCode = 999
)

// String returns a human-readable description of the error code.
func (e ErrorCode) String() string {
	switch e {
	case Success:
		return "success"
	case ErrInvalidValue:
		return "invalid value"
	case ErrOutOfMemory:
		return "out of memory"
	case ErrNotInitialized:
		return "not initialized"
	case ErrDeinitialized:
		return "deinitialized"
	case ErrNoDevice:
		return "no CUDA device found"
	case ErrInvalidDevice:
		return "invalid device"
	case ErrInvalidImage:
		return "invalid image"
	case ErrInvalidContext:
		return "invalid context"
	case ErrInvalidPTX:
		return "invalid PTX"
	case ErrFileNotFound:
		return "file not found"
	case ErrInvalidHandle:
		return "invalid handle"
	case ErrNotReady:
		return "not ready"
	case ErrIllegalAddress:
		return "illegal address"
	case ErrLaunchOutOfResources:
		return "launch out of resources"
	case ErrLaunchTimeout:
		return "launch timeout"
	case ErrLaunchFailed:
		return "launch failed"
	case ErrNotSupported:
		return "not supported"
	case ErrUnknown:
		return "unknown error"
	default:
		return fmt.Sprintf("error code %d", int(e))
	}
}

// ErrorCategory groups related errors for easier handling.
type ErrorCategory int

const (
	CategoryUnknown ErrorCategory = iota
	CategoryInit           // Initialization errors
	CategoryDevice         // Device-related errors
	CategoryMemory         // Memory allocation/access errors
	CategoryKernel         // Kernel execution errors
	CategoryPTX            // PTX compilation errors
	CategoryContext        // Context management errors
	CategoryDriver         // Driver-level errors
)

// Category returns the error category for an error code.
func (e ErrorCode) Category() ErrorCategory {
	switch e {
	case ErrNotInitialized, ErrDeinitialized:
		return CategoryInit
	case ErrNoDevice, ErrInvalidDevice, ErrDeviceNotLicensed:
		return CategoryDevice
	case ErrOutOfMemory, ErrIllegalAddress, ErrMisalignedAddress:
		return CategoryMemory
	case ErrLaunchOutOfResources, ErrLaunchTimeout, ErrLaunchFailed,
		ErrLaunchIncompatibleTexturing, ErrCooperativeLaunchTooLarge:
		return CategoryKernel
	case ErrInvalidPTX, ErrNoBinaryForGpu, ErrInvalidImage:
		return CategoryPTX
	case ErrInvalidContext, ErrContextAlreadyCurrent, ErrContextAlreadyInUse,
		ErrContextIsDestroyed:
		return CategoryContext
	case ErrOperatingSystem, ErrSharedObjectSymbolNotFound,
		ErrSharedObjectInitFailed:
		return CategoryDriver
	default:
		return CategoryUnknown
	}
}

// CUDAError provides detailed CUDA error information.
type CUDAError struct {
	Code     ErrorCode
	Context  string // Operation that failed
	Details  string // Additional details
	Cause    error  // Underlying error (if any)
}

func (e *CUDAError) Error() string {
	msg := fmt.Sprintf("CUDA %s: %s", e.Context, e.Code)
	if e.Details != "" {
		msg += fmt.Sprintf(" (%s)", e.Details)
	}
	if e.Cause != nil {
		msg += fmt.Sprintf(": %v", e.Cause)
	}
	return msg
}

// Unwrap returns the underlying error.
func (e *CUDAError) Unwrap() error {
	return e.Cause
}

// Is returns true if target matches this error.
func (e *CUDAError) Is(target error) bool {
	if t, ok := target.(*CUDAError); ok {
		return e.Code == t.Code
	}
	return false
}

// NewError creates a new CUDA error.
func NewError(code ErrorCode, context string) *CUDAError {
	return &CUDAError{
		Code:    code,
		Context: context,
	}
}

// WithDetails adds details to the error.
func (e *CUDAError) WithDetails(details string) *CUDAError {
	e.Details = details
	return e
}

// WithCause adds an underlying cause.
func (e *CUDAError) WithCause(cause error) *CUDAError {
	e.Cause = cause
	return e
}

// IsMemoryError returns true if the error is memory-related.
func IsMemoryError(err error) bool {
	if e, ok := err.(*CUDAError); ok {
		return e.Code.Category() == CategoryMemory
	}
	return false
}

// IsKernelError returns true if the error is kernel execution-related.
func IsKernelError(err error) bool {
	if e, ok := err.(*CUDAError); ok {
		return e.Code.Category() == CategoryKernel
	}
	return false
}

// IsPTXError returns true if the error is PTX compilation-related.
func IsPTXError(err error) bool {
	if e, ok := err.(*CUDAError); ok {
		return e.Code.Category() == CategoryPTX
	}
	return false
}

// IsDeviceError returns true if the error is device-related.
func IsDeviceError(err error) bool {
	if e, ok := err.(*CUDAError); ok {
		return e.Code.Category() == CategoryDevice
	}
	return false
}

// RecoverableError returns true if the error might be recoverable.
func RecoverableError(err error) bool {
	if e, ok := err.(*CUDAError); ok {
		switch e.Code {
		case ErrNotReady, ErrLaunchTimeout, ErrOutOfMemory:
			return true
		}
	}
	return false
}

// Suggestions returns possible actions to fix the error.
func Suggestions(err error) []string {
	e, ok := err.(*CUDAError)
	if !ok {
		return nil
	}

	switch e.Code {
	case ErrNoDevice:
		return []string{
			"Ensure NVIDIA GPU is installed",
			"Check that NVIDIA drivers are installed",
			"Verify GPU is visible with 'nvidia-smi'",
		}
	case ErrOutOfMemory:
		return []string{
			"Reduce batch size",
			"Use smaller model",
			"Free unused GPU memory",
			"Use gradient checkpointing",
		}
	case ErrInvalidPTX:
		return []string{
			"Check PTX syntax",
			"Verify target architecture matches GPU",
			"Update CUDA driver",
		}
	case ErrLaunchOutOfResources:
		return []string{
			"Reduce number of threads per block",
			"Reduce shared memory usage",
			"Reduce register usage",
		}
	case ErrLaunchTimeout:
		return []string{
			"Reduce kernel complexity",
			"Disable display timeout (TDR on Windows)",
			"Split kernel into smaller parts",
		}
	default:
		return nil
	}
}

// ErrorHandler is a callback for handling CUDA errors.
type ErrorHandler func(err *CUDAError)

// DefaultErrorHandler logs errors with suggestions.
var DefaultErrorHandler ErrorHandler = func(err *CUDAError) {
	// Could log to stderr or a logger
}

// Check converts a CUDA error code to an error, with context.
func Check(code int, context string) error {
	if code == 0 {
		return nil
	}
	err := &CUDAError{
		Code:    ErrorCode(code),
		Context: context,
	}
	if DefaultErrorHandler != nil {
		DefaultErrorHandler(err)
	}
	return err
}

// MustSucceed panics if the error is not nil.
func MustSucceed(err error) {
	if err != nil {
		panic(err)
	}
}

// Wrap wraps an error with CUDA context.
func Wrap(err error, context string) error {
	if err == nil {
		return nil
	}
	if e, ok := err.(*CUDAError); ok {
		return &CUDAError{
			Code:    e.Code,
			Context: context,
			Cause:   e,
		}
	}
	return &CUDAError{
		Code:    ErrUnknown,
		Context: context,
		Cause:   err,
	}
}
