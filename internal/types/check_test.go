package types

import (
	"strings"
	"testing"

	"github.com/esque-lang/esquec/internal/parse"
)

func check(t *testing.T, src string) (*CheckedFile, error) {
	t.Helper()
	f, err := parse.File("test.esq", []byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return Check(f)
}

func mustCheck(t *testing.T, src string) *CheckedFile {
	t.Helper()
	cf, err := check(t, src)
	if err != nil {
		t.Fatalf("check error: %v\nsrc:\n%s", err, src)
	}
	return cf
}

func mustReject(t *testing.T, src, wantSubstr string) {
	t.Helper()
	_, err := check(t, src)
	if err == nil {
		t.Fatalf("expected check error containing %q, got success\nsrc:\n%s", wantSubstr, src)
	}
	if wantSubstr != "" && !strings.Contains(err.Error(), wantSubstr) {
		t.Fatalf("error %q does not contain %q", err.Error(), wantSubstr)
	}
}

func TestCheckSimpleI32(t *testing.T) {
	cf := mustCheck(t, `fn main() -> i32 = 42`)
	if len(cf.Fns) != 1 {
		t.Fatalf("fns = %d", len(cf.Fns))
	}
	fn := cf.Fns[0]
	if fn.Name != "main" {
		t.Errorf("name = %q", fn.Name)
	}
	if !fn.RetType.Equal(I32()) {
		t.Errorf("ret = %s", fn.RetType)
	}
}

func TestCheckTensorReturn(t *testing.T) {
	cf := mustCheck(t, `fn k() -> f32[3] = [1.0, 2.0, 3.0]`)
	fn := cf.Fns[0]
	if !fn.RetType.IsTensor() {
		t.Fatalf("not tensor: %s", fn.RetType)
	}
	if fn.RetType.Rank() != 1 {
		t.Errorf("rank = %d", fn.RetType.Rank())
	}
	if fn.RetType.K != KF32 {
		t.Errorf("kind = %v", fn.RetType.K)
	}
}

func TestCheckGenericMonomorphizes(t *testing.T) {
	cf := mustCheck(t, `
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)
fn use_a(x: f32[4], y: f32[4]) -> f32 = dot(x, y)
fn use_b(x: f32[8], y: f32[8]) -> f32 = dot(x, y)
`)
	// Expect: dot__4, dot__8, use_a, use_b — at least these.
	got := map[string]bool{}
	for _, fn := range cf.Fns {
		got[fn.Name] = true
	}
	for _, want := range []string{"use_a", "use_b"} {
		if !got[want] {
			t.Errorf("missing fn %q", want)
		}
	}
	// Generic dot may appear under monomorphized names with shape values appended.
	hasMono4 := false
	hasMono8 := false
	for _, fn := range cf.Fns {
		if fn.IsMonomorphized && fn.OriginalName == "dot" {
			if fn.ShapeValues["N"] == 4 {
				hasMono4 = true
			}
			if fn.ShapeValues["N"] == 8 {
				hasMono8 = true
			}
		}
	}
	if !hasMono4 || !hasMono8 {
		t.Errorf("missing monomorphizations: dot__4=%v dot__8=%v", hasMono4, hasMono8)
	}
}

func TestCheckShapeMismatch(t *testing.T) {
	mustReject(t, `fn k(x: f32[3], y: f32[4]) -> f32[3] = x .+ y`, "")
}

func TestCheckTypeMismatch(t *testing.T) {
	mustReject(t, `fn k() -> i32 = 1.0`, "")
}

func TestCheckUnknownIdent(t *testing.T) {
	mustReject(t, `fn k() -> i32 = oops`, "")
}

func TestCheckIfElseTypeMatch(t *testing.T) {
	mustCheck(t, `fn k(x: i32) -> i32 = if x > 0 { 1 } else { -1 }`)
}

func TestCheckBoolType(t *testing.T) {
	cf := mustCheck(t, `fn k() -> bool = true`)
	if cf.Fns[0].RetType.K != KBool {
		t.Fatalf("ret = %s", cf.Fns[0].RetType)
	}
}

func TestTypeEqual(t *testing.T) {
	if !I32().Equal(I32()) {
		t.Error("i32 != i32")
	}
	if I32().Equal(I64()) {
		t.Error("i32 == i64")
	}
	a := F32().WithShape([]ShapeDim{ConcreteDim(3), ConcreteDim(4)})
	b := F32().WithShape([]ShapeDim{ConcreteDim(3), ConcreteDim(4)})
	if !a.Equal(b) {
		t.Error("equal tensors not equal")
	}
	c := F32().WithShape([]ShapeDim{ConcreteDim(3), ConcreteDim(5)})
	if a.Equal(c) {
		t.Error("unequal tensors equal")
	}
}

func TestTypeStringTensor(t *testing.T) {
	tt := F32().WithShape([]ShapeDim{ConcreteDim(3), VarDim("N")})
	if got, want := tt.String(), "f32[3, N]"; got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

// TestCheckRangeAccept: literal-bounded i32 range type-checks.
func TestCheckRangeAccept(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/(0..5)`)
	mustCheck(t, `fn k() -> i32 = +/(1..=5)`)
}

// TestCheckReduceOpsAccept: all four reduction operators type-check on
// numeric tensors (v0.14).
func TestCheckReduceOpsAccept(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/[10, 1, 2, 3]`)
	mustCheck(t, `fn k() -> i32 = -/[10, 1, 2, 3]`)
	mustCheck(t, `fn k() -> i32 = */[2, 3, 4]`)
	mustCheck(t, `fn k() -> i32 = //[120, 2, 3]`)
}

// TestCheckReduceRejectsBool: reductions require numeric element types.
// `-/` and `//` go through the same `IsNumeric()` gate as `+/` and
// `*/`; this test pins the rejection for the new operators so a future
// refactor can't quietly let a `bool[N]` reduction through.
func TestCheckReduceRejectsBool(t *testing.T) {
	mustReject(t, `fn k(v: bool[3]) -> bool = -/v`, "numeric")
	mustReject(t, `fn k(v: bool[3]) -> bool = //v`, "numeric")
}

// TestCheckReduceRejectsEmpty: a left-fold reduction has no seed value
// for an empty tensor. The typechecker already rejects empty / reversed
// ranges (see TestCheckRangeRejectEmpty); this test pins that the
// rejection extends to the new `-/` and `//` operators so an empty
// tensor can never reach CEIR via them.
func TestCheckReduceRejectsEmpty(t *testing.T) {
	mustReject(t, `fn k() -> i32 = -/(5..5)`, "empty")
	mustReject(t, `fn k() -> i32 = //(5..5)`, "empty")
}

// TestCheckRangeRejectNonConst: dynamic bounds rejected.
// In v0.11 const-foldable arithmetic is allowed (see TestCheckRangeConstArith);
// only references to runtime values like parameters remain rejected.
func TestCheckRangeRejectNonConst(t *testing.T) {
	mustReject(t, `fn k(n: i32) -> i32 = +/(0..n)`, "constant integer")
}

// TestCheckRangeConstArith: const-folded arithmetic in range bounds is
// accepted in v0.11.
func TestCheckRangeConstArith(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/(0..(2+3))`)
	mustCheck(t, `fn k() -> i32 = +/(0..(4*2))`)
	mustCheck(t, `fn k() -> i32 = +/((1+1)..(2*5))`)
}

// TestCheckRangeRejectEmpty: zero-length / reversed ranges rejected.
func TestCheckRangeRejectEmpty(t *testing.T) {
	mustReject(t, `fn k() -> i32 = +/(5..5)`, "empty")
	mustReject(t, `fn k() -> i32 = +/(5..3)`, "empty")
}

// TestCheckRangeRejectOversized: > 1<<20 elements rejected.
func TestCheckRangeRejectOversized(t *testing.T) {
	mustReject(t, `fn k() -> i32 = +/(0..2000000)`, "range too large")
}

// TestCheckTabulateAccept: 5-element i32 tabulate type-checks.
func TestCheckTabulateAccept(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/(tabulate(5, |i| i*2))`)
}

// TestCheckTabulateLargeNAccepted: tabulate counts above the unroll
// threshold (32) are accepted in v0.11 — they emit OpTabulateLoop in
// CEIR and a counted runtime loop in MIR.
func TestCheckTabulateLargeNAccepted(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/(tabulate(64, |i| i))`)
}

// TestCheckTabulateRejectNonConstN: non-const N rejected.
// In v0.11 const-foldable arithmetic in N is allowed (see
// TestCheckTabulateConstArithN); only references to runtime values like
// parameters remain rejected.
func TestCheckTabulateRejectNonConstN(t *testing.T) {
	mustReject(t, `fn k(n: i32) -> i32 = +/(tabulate(n, |i| i))`, "constant integer")
}

// TestCheckTabulateConstArithN: const-folded arithmetic in N is accepted
// in v0.11 (e.g. tabulate(2*16+1, ...)).
func TestCheckTabulateConstArithN(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/(tabulate(2*16+1, |i| i))`)
	mustCheck(t, `fn k() -> i32 = +/(tabulate(8*4, |i| i))`)
}

// TestCheckScanAccept.
func TestCheckScanAccept(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = +/(scan(0, |a, x| a + x, [1, 2, 3]))`)
}

// TestCheckScanRejectInitTypeMismatch.
func TestCheckScanRejectInitTypeMismatch(t *testing.T) {
	mustReject(t, `fn k() -> i32 = +/(scan(0.0, |a, x| a + x, [1, 2, 3]))`, "")
}

// TestCheckIterateUntilAccept.
func TestCheckIterateUntilAccept(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = iterate_until(0, |s| s + 1, |s| s == 5, 10)`)
}

// TestCheckIterateUntilRejectMaxOversized.
func TestCheckIterateUntilRejectMaxOversized(t *testing.T) {
	mustReject(t,
		`fn k() -> i32 = iterate_until(0, |s| s + 1, |s| s == 5, 64)`,
		"OpIterateUntilLoop")
}

// TestCheckIterateUntilRejectPredNotBool.
func TestCheckIterateUntilRejectPredNotBool(t *testing.T) {
	mustReject(t,
		`fn k() -> i32 = iterate_until(0, |s| s + 1, |s| s + 1, 10)`,
		"bool")
}

// TestCheckEachAccept: print_i32 over an i32 range type-checks when
// the enclosing function is `@io`.
func TestCheckEachAccept(t *testing.T) {
	mustCheck(t, `@io fn main() -> i32 = { each(0..3, print_i32); 0 }`)
}

// TestCheckEachAcceptsPureUserFn: as of v0.13 the legacy print-only
// allowlist is gone; any pure user function whose parameter matches
// the element type is accepted by `each`, even from a pure caller.
func TestCheckEachAcceptsPureUserFn(t *testing.T) {
	src := `
fn my_fn(x: i32) -> i32 = x
fn main() -> i32 = { each(0..3, my_fn); 0 }`
	mustCheck(t, src)
}

// TestCheckEachRejectsIOInPureCaller: `each(v, print_i32)` from a
// pure `main` is rejected because print_i32 carries `@io` and the
// caller does not propagate that effect.
func TestCheckEachRejectsIOInPureCaller(t *testing.T) {
	mustReject(t,
		`fn main() -> i32 = { each(0..3, print_i32); 0 }`,
		"@io")
}

// TestCheckEachRejectElemTypeMismatch: print_i32 over an f32 tensor
// fails the element-type check (with @io on the caller so the effect
// pre-check passes and we reach the element-type comparison).
func TestCheckEachRejectElemTypeMismatch(t *testing.T) {
	mustReject(t, `@io fn main() -> i32 = { each([1.0, 2.0, 3.0], print_i32); 0 }`, "")
}

// TestCheckCharLiteral: char literal type-checks as i32.
func TestCheckCharLiteral(t *testing.T) {
	cf := mustCheck(t, `fn main() -> i32 = 'A'`)
	if !cf.Fns[0].RetType.Equal(I32()) {
		t.Errorf("ret = %s", cf.Fns[0].RetType)
	}
}

// TestCheckIntSuffixI32: explicit _i32 type-checks as i32.
func TestCheckIntSuffixI32(t *testing.T) {
	mustCheck(t, `fn main() -> i32 = 42_i32`)
}

// TestCheckIntSuffixI64Accepted: in v0.11 _i64 type-checks and codegen
// can lower it as a 64-bit GPR value.
func TestCheckIntSuffixI64Accepted(t *testing.T) {
	cf := mustCheck(t, `fn main() -> i64 = 42_i64`)
	if !cf.Fns[0].RetType.Equal(I64()) {
		t.Errorf("ret = %s, want i64", cf.Fns[0].RetType)
	}
}

// TestCheckIntSuffixU32Accepted: in v0.11 _u32 type-checks and codegen
// lowers it as a 64-bit GPR value (with unsigned semantics for div/cmp).
func TestCheckIntSuffixU32Accepted(t *testing.T) {
	cf := mustCheck(t, `fn main() -> u32 = 7_u32`)
	if !cf.Fns[0].RetType.Equal(U32()) {
		t.Errorf("ret = %s, want u32", cf.Fns[0].RetType)
	}
}

// TestCheckIntSuffixI8Accepted: in v0.12 _i8 type-checks and codegen
// lowers it as a 64-bit GPR value (with sign-canonicalised low byte).
func TestCheckIntSuffixI8Accepted(t *testing.T) {
	cf := mustCheck(t, `fn main() -> i8 = 42_i8`)
	if !cf.Fns[0].RetType.Equal(I8()) {
		t.Errorf("ret = %s, want i8", cf.Fns[0].RetType)
	}
}

// TestCheckIntSuffixI8Negative: -1_i8 type-checks (unary minus over
// the i8 literal 1).
func TestCheckIntSuffixI8Negative(t *testing.T) {
	mustCheck(t, `fn main() -> i8 = -1_i8`)
}

// TestCheckIntSuffixI8Overflow: 128 does not fit in i8.
func TestCheckIntSuffixI8Overflow(t *testing.T) {
	mustReject(t, `fn main() -> i8 = 128_i8`, "i8")
}

// TestCheckIntSuffixU8Accepted: in v0.12 _u8 type-checks and codegen
// lowers it as a 64-bit GPR value (with zero-canonicalised low byte).
func TestCheckIntSuffixU8Accepted(t *testing.T) {
	cf := mustCheck(t, `fn main() -> u8 = 42_u8`)
	if !cf.Fns[0].RetType.Equal(U8()) {
		t.Errorf("ret = %s, want u8", cf.Fns[0].RetType)
	}
}

// TestCheckIntSuffixU8Max: 255_u8 is the largest valid u8.
func TestCheckIntSuffixU8Max(t *testing.T) {
	mustCheck(t, `fn main() -> u8 = 255_u8`)
}

// TestCheckIntSuffixU8Overflow: 256 does not fit in u8.
func TestCheckIntSuffixU8Overflow(t *testing.T) {
	mustReject(t, `fn main() -> u8 = 256_u8`, "u8")
}

// TestCheckIntSuffixI16Rejected: _i16 still rejected; codegen support
// is planned for a future milestone.
func TestCheckIntSuffixI16Rejected(t *testing.T) {
	mustReject(t, `fn main() -> i32 = 7_i16`, "_i16")
}

// TestCheckStringLit: a "..." literal type-checks as `string` and can
// be passed to print_str (the only string-consuming intrinsic) from
// an `@io` caller.
func TestCheckStringLit(t *testing.T) {
	mustCheck(t, `@io fn main() -> i32 = { print_str("hello"); 0 }`)
}

// TestCheckRejectStringConcat: string concatenation is still rejected
// pending the runtime allocator. The caller is annotated `@io` so
// that we exercise the concat error rather than the effect error.
func TestCheckRejectStringConcat(t *testing.T) {
	mustReject(t, `@io fn main() -> i32 = { print_str("a" + "b"); 0 }`, "string")
}

// TestCheckIOEffectPropagates: an `@io` callee invoked from a pure
// caller is rejected; annotating the caller `@io` lets the call
// type-check.
func TestCheckIOEffectPropagates(t *testing.T) {
	mustReject(t, `fn main() -> i32 = print_i32(1)`, "@io")
	mustCheck(t, `@io fn main() -> i32 = print_i32(1)`)
}

// TestCheckIORequiresAnnotationOnIntermediate: an `@io` helper that
// transitively calls print_i32 must itself be `@io`. Without the
// annotation on `helper`, the inner print_i32 call is rejected even
// if `main` is `@io`.
func TestCheckIORequiresAnnotationOnIntermediate(t *testing.T) {
	bad := `
fn helper() -> i32 = print_i32(1)
@io fn main() -> i32 = helper()`
	mustReject(t, bad, "@io")

	good := `
@io fn helper() -> i32 = print_i32(1)
@io fn main() -> i32 = helper()`
	mustCheck(t, good)
}

// TestCheckRejectStringParam: function parameters of type `string`
// require a multi-register call ABI that v0.12 does not implement.
func TestCheckRejectStringParam(t *testing.T) {
	mustReject(t, `fn k(s: string) -> i32 = 0`, "string")
}

// TestCheckRejectStringReturn: returning a string from a user-defined
// function is reserved for the same reason.
func TestCheckRejectStringReturn(t *testing.T) {
	mustReject(t, `fn k() -> string = "y"`, "string")
}

// TestCheckIntLiteralOverflow: too-large unsuffixed literal hints at
// the _i64 suffix.
func TestCheckIntLiteralOverflow(t *testing.T) {
	mustReject(t, `fn main() -> i32 = 4000000000`, "i32")
}

// TestCheckLetIn: bare `let x = e in body` desugars to a Block
// statement, so the existing block scoping rules apply.
func TestCheckLetIn(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = let x = 17 in x + 25`)
}

// TestCheckLetInNested: right-associative let-in chains.
func TestCheckLetInNested(t *testing.T) {
	mustCheck(t, `fn k() -> i32 =
        let a = 10 in
        let b = a * 2 in
        a + b`)
}

// TestCheckLetInTypeAnnotation: explicit type annotation is honored.
func TestCheckLetInTypeAnnotation(t *testing.T) {
	mustCheck(t, `fn k() -> i32 = let x: i32 = 5 in x * 2`)
}

// TestCheckLetInScopeLeak: the bound name is not visible after the
// let-in body completes (here the let-in is itself the function body
// so there is no "after", but a sibling fn must not see x).
func TestCheckLetInScopeLeak(t *testing.T) {
	mustReject(t, `
        fn k() -> i32 = let x = 1 in x
        fn m() -> i32 = x
    `, "unknown identifier")
}

// TestCheckLetInTypeMismatch: init type must match annotation.
func TestCheckLetInTypeMismatch(t *testing.T) {
	mustReject(t, `fn k() -> i32 = let x: i32 = 1.0 in x`, "")
}

func TestInstKey(t *testing.T) {
	inst := &Instantiation{
		FnName:      "dot",
		ShapeValues: map[string]int64{"N": 8, "M": 16},
	}
	got := inst.InstKey()
	// Keys sort alphabetically: M then N.
	if got != "dot__16_8" {
		t.Errorf("InstKey = %q want %q", got, "dot__16_8")
	}
}
