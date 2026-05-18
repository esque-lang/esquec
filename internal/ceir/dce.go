package ceir

// hasSideEffects reports whether an op has effects other than producing
// its SSA result. The DCE pass keeps these instructions even when their
// result is unused.
func hasSideEffects(op Op) bool {
	switch op {
	case OpCall, OpScatter:
		return true
	}
	return false
}

// DCE removes instructions whose SSA result is never used and which have
// no side effects. It iterates to a fixed point because removing one
// instruction can render a previously-live operand dead.
func DCE(f *Func) {
	for {
		used := map[int]bool{}
		used[f.Result.ID] = true
		// Mark used: scan in reverse so we propagate uses upward.
		for i := len(f.Body) - 1; i >= 0; i-- {
			in := f.Body[i]
			if used[in.Result.ID] || hasSideEffects(in.Op) {
				for _, a := range in.Args {
					used[a.ID] = true
				}
				// nested loop bodies: scan their args too
				for _, b := range in.BodyInsts {
					for _, a := range b.Args {
						used[a.ID] = true
					}
				}
			}
		}
		// Sweep.
		out := make([]Inst, 0, len(f.Body))
		for _, in := range f.Body {
			if used[in.Result.ID] || hasSideEffects(in.Op) {
				out = append(out, in)
			}
		}
		if len(out) == len(f.Body) {
			f.Body = out
			return
		}
		f.Body = out
	}
}
