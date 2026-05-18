package ceir

// ConstFold performs simple constant folding on a function.
func ConstFold(f *Func) {
	constInts := map[int]int64{}
	constBools := map[int]bool{}
	out := make([]Inst, 0, len(f.Body))

	for _, in := range f.Body {
		switch in.Op {
		case OpConstInt:
			constInts[in.Result.ID] = in.Imm
			out = append(out, in)

		case OpConstBool:
			constBools[in.Result.ID] = in.ImmB
			out = append(out, in)

		case OpAdd, OpSub, OpMul, OpDiv, OpMod:
			a, aok := constInts[in.Args[0].ID]
			b, bok := constInts[in.Args[1].ID]
			if aok && bok {
				var v int64
				switch in.Op {
				case OpAdd:
					v = a + b
				case OpSub:
					v = a - b
				case OpMul:
					v = a * b
				case OpDiv:
					if b != 0 {
						v = a / b
					}
				case OpMod:
					if b != 0 {
						v = a % b
					}
				}
				constInts[in.Result.ID] = v
				out = append(out, Inst{Result: in.Result, Op: OpConstInt, Imm: v})
				continue
			}
			out = append(out, in)

		case OpNeg:
			a, aok := constInts[in.Args[0].ID]
			if aok {
				v := -a
				constInts[in.Result.ID] = v
				out = append(out, Inst{Result: in.Result, Op: OpConstInt, Imm: v})
				continue
			}
			out = append(out, in)

		case OpNot:
			a, aok := constBools[in.Args[0].ID]
			if aok {
				v := !a
				constBools[in.Result.ID] = v
				out = append(out, Inst{Result: in.Result, Op: OpConstBool, ImmB: v})
				continue
			}
			out = append(out, in)

		case OpEq, OpNe, OpLt, OpLe, OpGt, OpGe:
			a, aok := constInts[in.Args[0].ID]
			b, bok := constInts[in.Args[1].ID]
			if aok && bok {
				var v bool
				switch in.Op {
				case OpEq:
					v = a == b
				case OpNe:
					v = a != b
				case OpLt:
					v = a < b
				case OpLe:
					v = a <= b
				case OpGt:
					v = a > b
				case OpGe:
					v = a >= b
				}
				constBools[in.Result.ID] = v
				out = append(out, Inst{Result: in.Result, Op: OpConstBool, ImmB: v})
				continue
			}
			out = append(out, in)

		case OpAnd, OpOr:
			a, aok := constBools[in.Args[0].ID]
			b, bok := constBools[in.Args[1].ID]
			if aok && bok {
				var v bool
				switch in.Op {
				case OpAnd:
					v = a && b
				case OpOr:
					v = a || b
				}
				constBools[in.Result.ID] = v
				out = append(out, Inst{Result: in.Result, Op: OpConstBool, ImmB: v})
				continue
			}
			out = append(out, in)

		case OpSelect:
			c, cok := constBools[in.Args[0].ID]
			if cok {
				// select(true, t, f) -> copy t; select(false, t, f) -> copy f
				var picked Value
				if c {
					picked = in.Args[1]
				} else {
					picked = in.Args[2]
				}
				out = append(out, Inst{Result: in.Result, Op: OpCopy, Args: []Value{picked}})
				// Propagate constants if possible
				if v, ok := constInts[picked.ID]; ok {
					constInts[in.Result.ID] = v
				}
				if v, ok := constBools[picked.ID]; ok {
					constBools[in.Result.ID] = v
				}
				continue
			}
			out = append(out, in)

		default:
			out = append(out, in)
		}
	}
	f.Body = out
}
