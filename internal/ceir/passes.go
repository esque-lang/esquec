package ceir

// Pass is a CEIR optimization pass that mutates a function in place.
type Pass interface {
	Name() string
	Run(*Func)
}

type passFn struct {
	name string
	run  func(*Func)
}

func (p *passFn) Name() string  { return p.name }
func (p *passFn) Run(f *Func)   { p.run(f) }

// PassFunc wraps a plain function as a Pass.
func PassFunc(name string, run func(*Func)) Pass {
	return &passFn{name: name, run: run}
}

// DefaultPipeline is a recommended order: const-fold, simplify, CSE, DCE.
// Const-fold first so simplify can recognise the resulting literals,
// simplify next so CSE sees the canonical forms, DCE last so it can
// remove copies introduced by CSE and Simplify.
func DefaultPipeline() []Pass {
	return []Pass{
		PassFunc("const-fold", ConstFold),
		PassFunc("simplify", Simplify),
		PassFunc("cse", CSE),
		PassFunc("dce", DCE),
	}
}

// RunPasses runs the given passes over every function in m. Passes mutate
// in place; the module's function pointers are unchanged.
func RunPasses(m *Module, passes []Pass) {
	for _, fn := range m.Fns {
		for _, p := range passes {
			p.Run(fn)
		}
	}
}
