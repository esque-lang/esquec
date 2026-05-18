package ceir

import "fmt"

// cseKey returns a canonical hash key for a pure instruction. Returns ""
// for instructions that are not safe to commonise (calls, loops, etc.).
func cseKey(in Inst) string {
	switch in.Op {
	case OpCall, OpIterateLoop, OpIterate, OpFold, OpFoldInit, OpScatter,
		OpGather, OpTensorLit,
		OpTabulateLoop, OpScanLoop, OpEachLoop:
		return ""
	}
	switch in.Op {
	case OpConstInt:
		return fmt.Sprintf("ci:%d:%d", in.Result.Type.K, in.Imm)
	case OpConstFloat:
		// Bits to keep -0 vs 0 distinct.
		return fmt.Sprintf("cf:%d:%v", in.Result.Type.K, in.ImmF)
	case OpConstBool:
		return fmt.Sprintf("cb:%v", in.ImmB)
	}
	// Generic op with operands.
	args := make([]byte, 0, 16)
	for _, a := range in.Args {
		args = fmt.Appendf(args, "%d,", a.ID)
	}
	return fmt.Sprintf("op:%d:%d:%s", in.Op, in.Result.Type.K, string(args))
}

// CSE performs common-subexpression elimination on a function. Instructions
// with identical keys keep their first occurrence and later occurrences
// become OpCopy of the earlier SSA value. Subsequent DCE will remove the
// copies if their results aren't otherwise used.
func CSE(f *Func) {
	seen := map[string]Value{}  // key -> first-defining value
	rewrite := map[int]int{}    // dead-id -> live-id
	out := make([]Inst, 0, len(f.Body))
	rename := func(v Value) Value {
		if id, ok := rewrite[v.ID]; ok {
			v.ID = id
		}
		return v
	}
	for _, in := range f.Body {
		// Rewrite operands first so equal subexpressions hash equally.
		for i := range in.Args {
			in.Args[i] = rename(in.Args[i])
		}
		key := cseKey(in)
		if key == "" {
			out = append(out, in)
			continue
		}
		if first, ok := seen[key]; ok {
			rewrite[in.Result.ID] = first.ID
			out = append(out, Inst{
				Result: in.Result,
				Op:     OpCopy,
				Args:   []Value{first},
			})
			continue
		}
		seen[key] = in.Result
		out = append(out, in)
	}
	// Ensure final result reflects renames too.
	if id, ok := rewrite[f.Result.ID]; ok {
		f.Result.ID = id
	}
	f.Body = out
}
