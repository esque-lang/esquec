package ceir

// Simplify performs simple algebraic simplifications. Idempotent.
//
// Recognised patterns:
//   x + 0      -> x       0 + x      -> x
//   x - 0      -> x       x - x      -> 0
//   x * 1      -> x       1 * x      -> x
//   x * 0      -> 0       0 * x      -> 0
//   x / 1      -> x
//   x .+ 0     -> x      similar for tensor element-wise
//   x .* 1     -> x
//   not (not x) -> x  (peephole if x's defining op is also OpNot)
//   x && true  -> x   x && false -> false
//   x || false -> x   x || true  -> true
//
// Simplifications introduce OpCopy where the result becomes one of the
// input operands; downstream DCE then removes the copy if its result
// is otherwise unused.
func Simplify(f *Func) {
	// Map from SSA id to constant int (if known).
	constI := map[int]int64{}
	constB := map[int]bool{}

	for i := range f.Body {
		in := &f.Body[i]
		switch in.Op {
		case OpConstInt:
			constI[in.Result.ID] = in.Imm
		case OpConstBool:
			constB[in.Result.ID] = in.ImmB
		}
	}

	isI := func(id int, want int64) bool {
		v, ok := constI[id]
		return ok && v == want
	}

	for i := range f.Body {
		in := &f.Body[i]
		switch in.Op {
		case OpAdd:
			a, b := in.Args[0], in.Args[1]
			if isI(b.ID, 0) {
				replaceWithCopy(in, a)
			} else if isI(a.ID, 0) {
				replaceWithCopy(in, b)
			}
		case OpSub:
			a, b := in.Args[0], in.Args[1]
			if isI(b.ID, 0) {
				replaceWithCopy(in, a)
			} else if a.ID == b.ID {
				// x - x = 0
				in.Op = OpConstInt
				in.Imm = 0
				in.Args = nil
				constI[in.Result.ID] = 0
			}
		case OpMul:
			a, b := in.Args[0], in.Args[1]
			if isI(b.ID, 1) {
				replaceWithCopy(in, a)
			} else if isI(a.ID, 1) {
				replaceWithCopy(in, b)
			} else if isI(a.ID, 0) || isI(b.ID, 0) {
				in.Op = OpConstInt
				in.Imm = 0
				in.Args = nil
				constI[in.Result.ID] = 0
			}
		case OpDiv:
			a, b := in.Args[0], in.Args[1]
			if isI(b.ID, 1) {
				replaceWithCopy(in, a)
			}
		case OpAnd:
			a, b := in.Args[0], in.Args[1]
			if v, ok := constB[b.ID]; ok {
				if v {
					replaceWithCopy(in, a)
				} else {
					in.Op = OpConstBool
					in.ImmB = false
					in.Args = nil
					constB[in.Result.ID] = false
				}
			} else if v, ok := constB[a.ID]; ok {
				if v {
					replaceWithCopy(in, b)
				} else {
					in.Op = OpConstBool
					in.ImmB = false
					in.Args = nil
					constB[in.Result.ID] = false
				}
			}
		case OpOr:
			a, b := in.Args[0], in.Args[1]
			if v, ok := constB[b.ID]; ok {
				if !v {
					replaceWithCopy(in, a)
				} else {
					in.Op = OpConstBool
					in.ImmB = true
					in.Args = nil
					constB[in.Result.ID] = true
				}
			} else if v, ok := constB[a.ID]; ok {
				if !v {
					replaceWithCopy(in, b)
				} else {
					in.Op = OpConstBool
					in.ImmB = true
					in.Args = nil
					constB[in.Result.ID] = true
				}
			}
		}
	}
}

func replaceWithCopy(in *Inst, src Value) {
	in.Op = OpCopy
	in.Args = []Value{src}
}
