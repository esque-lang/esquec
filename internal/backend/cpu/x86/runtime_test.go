package x86

import "testing"

// TestBuildPrintI32Shape sanity-checks the synthesized print_i32
// runtime function. It doesn't execute the code (that's the e2e
// test's job); it asserts the function looks well-formed: the right
// name, a non-trivial body, and a single trailing RET (0xC3) so we
// know the epilogue is in place.
func TestBuildPrintI32Shape(t *testing.T) {
	fn := BuildPrintI32()
	if fn.Name != "print_i32" {
		t.Errorf("Name = %q, want %q", fn.Name, "print_i32")
	}
	if len(fn.Code) < 64 {
		// Itoa loop + write syscall + epilogue should be well over 64
		// bytes; if we're below this the encoder is probably emitting
		// an empty body.
		t.Errorf("Code length = %d, want >= 64", len(fn.Code))
	}
	if last := fn.Code[len(fn.Code)-1]; last != 0xC3 {
		t.Errorf("last byte = 0x%02x, want 0xC3 (RET)", last)
	}
	if len(fn.Rels) != 0 {
		t.Errorf("Rels = %d, want 0 (the runtime is self-contained)", len(fn.Rels))
	}
}

// TestBuildPrintF32Shape sanity-checks the synthesized print_f32
// runtime function. Like the i32 shape test it asserts a well-formed
// body and trailing RET; print_f32 is meaningfully larger than
// print_i32 (sign handling + integer/fractional digit loops + the
// 1e6 scaling constant), so the lower bound is tighter.
func TestBuildPrintF32Shape(t *testing.T) {
	fn := BuildPrintF32()
	if fn.Name != "print_f32" {
		t.Errorf("Name = %q, want %q", fn.Name, "print_f32")
	}
	if len(fn.Code) < 128 {
		t.Errorf("Code length = %d, want >= 128", len(fn.Code))
	}
	if last := fn.Code[len(fn.Code)-1]; last != 0xC3 {
		t.Errorf("last byte = 0x%02x, want 0xC3 (RET)", last)
	}
	if len(fn.Rels) != 0 {
		t.Errorf("Rels = %d, want 0 (the runtime is self-contained)", len(fn.Rels))
	}
}

// TestIsPrintBuiltin spot-checks the dispatch table the build driver
// uses to decide whether to synthesize a runtime body.
func TestIsPrintBuiltin(t *testing.T) {
	if !IsPrintBuiltin("print_i32") {
		t.Errorf("print_i32 should be a runtime builtin")
	}
	if !IsPrintBuiltin("print_f32") {
		t.Errorf("print_f32 should be a runtime builtin")
	}
	if IsPrintBuiltin("user_function") {
		t.Errorf("user_function should not be a runtime builtin")
	}
}
