// Command esquec is the esque compiler driver.
//
// v0.0 supports a single subcommand:
//
//	esquec build <input.esq> -o <output>
//
// which compiles the source to a runnable ELF executable on x86-64 Linux.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/esque-lang/esquec/internal/backend/cpu/x86"
	"github.com/esque-lang/esquec/internal/ceir"
	"github.com/esque-lang/esquec/internal/diag"
	"github.com/esque-lang/esquec/internal/link"
	"github.com/esque-lang/esquec/internal/mir"
	"github.com/esque-lang/esquec/internal/object/elf"
	"github.com/esque-lang/esquec/internal/parse"
	"github.com/esque-lang/esquec/internal/types"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "build":
		os.Exit(buildCmd(os.Args[2:]))
	case "check":
		os.Exit(checkCmd(os.Args[2:]))
	case "version":
		fmt.Println("esquec 0.0")
		os.Exit(0)
	case "-h", "--help":
		usage()
		os.Exit(0)
	default:
		fmt.Fprintf(os.Stderr, "esquec: unknown subcommand %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: esquec <build|check|version> [args...]")
	fmt.Fprintln(os.Stderr, "  esquec build <input.esq> [-o <output>] [--emit=ast|ceir|mir|asm|obj]")
	fmt.Fprintln(os.Stderr, "  esquec check <input.esq>")
}

func buildCmd(args []string) int {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	out := fs.String("o", "", "output executable path")
	emit := fs.String("emit", "exe", "emit stage: exe|obj|asm|ceir|mir|ast")
	keepObj := fs.Bool("keep-obj", false, "keep intermediate .o file")
	if err := fs.Parse(reorderFlags(args, map[string]bool{"o": true, "emit": true})); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "build: exactly one input file is required")
		return 2
	}
	input := fs.Arg(0)
	if *out == "" {
		base := filepath.Base(input)
		ext := filepath.Ext(base)
		*out = base[:len(base)-len(ext)]
		if *out == base {
			*out = base + ".out"
		}
	}

	src, err := os.ReadFile(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "esquec: %v\n", err)
		return 1
	}

	// Front end
	file, err := parse.File(input, src)
	if err != nil {
		reportErr(err, src)
		return 1
	}
	if *emit == "ast" {
		fmt.Printf("%#v\n", file)
		return 0
	}

	checked, err := types.Check(file)
	if err != nil {
		reportErr(err, src)
		return 1
	}

	// Middle end
	ceirMod, err := ceir.LowerFromChecked(checked)
	if err != nil {
		reportErr(err, src)
		return 1
	}
	for _, fn := range ceirMod.Fns {
		ceir.ConstFold(fn)
	}
	if *emit == "ceir" {
		printCEIR(ceirMod)
		return 0
	}

	mirMod := mir.LowerFromCEIR(ceirMod)
	if *emit == "mir" {
		fmt.Printf("%#v\n", mirMod)
		return 0
	}

	// Back end
	compiled, err := x86.CompileModule(mirMod)
	if err != nil {
		reportErr(err, src)
		return 1
	}

	// Pull in runtime intrinsic bodies on demand. We scan the freshly
	// compiled functions for unresolved call relocations targeting
	// known runtime symbols and append synthesized CompiledFns for
	// those. If the user defined a function with the same name, the
	// symbol is already present and we leave it alone.
	compiled = appendRuntimeFns(compiled)

	if *emit == "asm" {
		for _, fn := range compiled {
			fmt.Printf("# %s\n", fn.Name)
			fmt.Printf("%x\n", fn.Code)
		}
		return 0
	}

	// Build ELF object
	obj := buildObject(compiled, mirMod.Rodata)
	objBytes, err := obj.Write()
	if err != nil {
		fmt.Fprintf(os.Stderr, "esquec: elf write: %v\n", err)
		return 1
	}

	objPath := *out + ".o"
	if *emit == "obj" {
		if err := os.WriteFile(*out, objBytes, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "esquec: %v\n", err)
			return 1
		}
		return 0
	}
	if err := os.WriteFile(objPath, objBytes, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "esquec: %v\n", err)
		return 1
	}
	if !*keepObj {
		defer os.Remove(objPath)
	}

	// Link
	if err := link.Link(*out, []string{objPath}, "-e", "_start"); err != nil {
		fmt.Fprintf(os.Stderr, "esquec: %v\n", err)
		return 1
	}
	return 0
}

func checkCmd(args []string) int {
	fs := flag.NewFlagSet("check", flag.ContinueOnError)
	if err := fs.Parse(reorderFlags(args, nil)); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "check: exactly one input file is required")
		return 2
	}
	input := fs.Arg(0)
	src, err := os.ReadFile(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "esquec: %v\n", err)
		return 1
	}
	file, err := parse.File(input, src)
	if err != nil {
		reportErr(err, src)
		return 1
	}
	if _, err := types.Check(file); err != nil {
		reportErr(err, src)
		return 1
	}
	return 0
}

// reorderFlags moves flag-like args ahead of positional args so that
// Go's flag package (which stops at the first non-flag token) accepts
// invocations like `esquec build foo.esq -o out`. valueFlags lists
// flag names that take a separate value argument (e.g. `-o out`); flags
// using `--name=value` or boolean flags don't need to appear there.
func reorderFlags(args []string, valueFlags map[string]bool) []string {
	flagsOut := make([]string, 0, len(args))
	posOut := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			posOut = append(posOut, args[i+1:]...)
			break
		}
		if len(a) > 1 && a[0] == '-' {
			flagsOut = append(flagsOut, a)
			name := a[1:]
			if len(name) > 0 && name[0] == '-' {
				name = name[1:]
			}
			// `-o=foo` / `--emit=ast` carry their value inline.
			hasInlineValue := false
			for j := 0; j < len(name); j++ {
				if name[j] == '=' {
					name = name[:j]
					hasInlineValue = true
					break
				}
			}
			if !hasInlineValue && valueFlags[name] && i+1 < len(args) {
				flagsOut = append(flagsOut, args[i+1])
				i++
			}
			continue
		}
		posOut = append(posOut, a)
	}
	return append(flagsOut, posOut...)
}

// reportErr writes an error to stderr. Structured *diag.Diagnostic
// errors are rendered with source context (file:line, source line, and
// caret highlight). Other errors fall back to plain printing.
func reportErr(err error, src []byte) {
	var d *diag.Diagnostic
	if errors.As(err, &d) {
		fmt.Fprint(os.Stderr, d.Render(src))
		return
	}
	fmt.Fprintf(os.Stderr, "%v\n", err)
}

// appendRuntimeFns scans `fns` for relocations referencing runtime
// intrinsics (e.g. `print_i32`) that aren't already defined in the
// module, and appends synthesized CompiledFns for those. Symbols the
// user defined themselves take precedence — we never overwrite them.
func appendRuntimeFns(fns []*x86.CompiledFn) []*x86.CompiledFn {
	defined := make(map[string]bool, len(fns))
	for _, fn := range fns {
		defined[fn.Name] = true
	}
	needed := map[string]bool{}
	for _, fn := range fns {
		for _, r := range fn.Rels {
			if !defined[r.Symbol] && x86.IsPrintBuiltin(r.Symbol) {
				needed[r.Symbol] = true
			}
		}
	}
	for sym := range needed {
		switch sym {
		case "print_i32":
			fns = append(fns, x86.BuildPrintI32())
			defined["print_i32"] = true
		case "print_f32":
			fns = append(fns, x86.BuildPrintF32())
			defined["print_f32"] = true
		case "print_str":
			fns = append(fns, x86.BuildPrintStr())
			defined["print_str"] = true
		}
	}
	return fns
}

// buildObject assembles the ELF object from compiled functions and
// optional .rodata blobs.
//
// Each function lives consecutively in .text. We emit each function
// name as a global symbol pointing at its offset in .text. The
// designated entry function is also exported as `_start`.
//
// Rodata entries are concatenated into a single .rodata section
// (respecting per-entry alignment) and exported as local STT_OBJECT
// symbols so that R_X86_64_PC32 references from .text resolve at link
// time.
func buildObject(fns []*x86.CompiledFn, rodata []mir.RodataEntry) *elf.Object {
	obj := &elf.Object{}

	type placed struct {
		fn     *x86.CompiledFn
		offset uint64
	}
	var placedFns []placed
	var text []byte
	for _, fn := range fns {
		off := uint64(len(text))
		text = append(text, fn.Code...)
		placedFns = append(placedFns, placed{fn: fn, offset: off})
	}

	obj.Add(elf.Section{
		Name:  ".text",
		Type:  elf.SHT_PROGBITS,
		Flags: elf.SHF_ALLOC | elf.SHF_EXECINSTR,
		Data:  text,
		Align: 16,
	})

	// .rodata: emit only when needed. We pad each entry to its
	// requested alignment (default 16) so SIMD loads at +offset stay
	// aligned. Each entry gets a local symbol pointing at its offset.
	if len(rodata) > 0 {
		var rodataBytes []byte
		type rodataSym struct {
			name   string
			offset uint64
			size   uint64
		}
		var rsyms []rodataSym
		for _, e := range rodata {
			align := e.Align
			if align == 0 {
				align = 1
			}
			if pad := uint64(len(rodataBytes)) % align; pad != 0 {
				rodataBytes = append(rodataBytes, make([]byte, align-pad)...)
			}
			rsyms = append(rsyms, rodataSym{
				name:   e.Sym,
				offset: uint64(len(rodataBytes)),
				size:   uint64(len(e.Bytes)),
			})
			rodataBytes = append(rodataBytes, e.Bytes...)
		}
		obj.Add(elf.Section{
			Name:  ".rodata",
			Type:  elf.SHT_PROGBITS,
			Flags: elf.SHF_ALLOC,
			Data:  rodataBytes,
			Align: 16,
		})
		for _, rs := range rsyms {
			obj.Symbols = append(obj.Symbols, elf.Symbol{
				Name:    rs.name,
				Section: ".rodata",
				Value:   rs.offset,
				Size:    rs.size,
				Type:    elf.STT_OBJECT,
				Bind:    elf.STB_LOCAL,
			})
		}
	}

	obj.Add(elf.Section{
		Name:  ".note.GNU-stack",
		Type:  elf.SHT_PROGBITS,
		Flags: 0,
		Data:  nil,
		Align: 1,
	})

	for _, p := range placedFns {
		obj.Symbols = append(obj.Symbols, elf.Symbol{
			Name:    p.fn.Name,
			Section: ".text",
			Value:   p.offset,
			Size:    uint64(len(p.fn.Code)),
			Type:    elf.STT_FUNC,
			Bind:    elf.STB_GLOBAL,
		})
		if p.fn.IsEntry {
			obj.Symbols = append(obj.Symbols, elf.Symbol{
				Name:    "_start",
				Section: ".text",
				Value:   p.offset,
				Size:    uint64(len(p.fn.Code)),
				Type:    elf.STT_FUNC,
				Bind:    elf.STB_GLOBAL,
			})
		}
	}

	// Translate per-function relocations into module relocations,
	// shifting offsets by where the function was placed in .text.
	for _, p := range placedFns {
		for _, r := range p.fn.Rels {
			var typ uint32
			switch r.Kind {
			case x86.RelPC32:
				typ = elf.R_X86_64_PLT32
			default:
				typ = elf.R_X86_64_PC32
			}
			obj.Relocs = append(obj.Relocs, elf.Reloc{
				ApplyTo: ".text",
				Offset:  p.offset + uint64(r.Offset),
				Sym:     r.Symbol,
				Type:    typ,
				Addend:  r.Addend,
			})
		}
	}
	return obj
}

func printCEIR(m *ceir.Module) {
	for _, fn := range m.Fns {
		fmt.Printf("fn %s(", fn.Name)
		for i, p := range fn.Params {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%s: %s", p.Name, p.Type)
		}
		fmt.Printf(") -> %s {\n", fn.RetType)
		for _, in := range fn.Body {
			switch in.Op {
			case ceir.OpConstInt:
				fmt.Printf("  v%d = const %d\n", in.Result.ID, in.Imm)
			case ceir.OpConstBool:
				fmt.Printf("  v%d = const %t\n", in.Result.ID, in.ImmB)
			case ceir.OpCall:
				fmt.Printf("  v%d = call %s(", in.Result.ID, in.FnName)
				for i, a := range in.Args {
					if i > 0 {
						fmt.Print(", ")
					}
					fmt.Printf("v%d", a.ID)
				}
				fmt.Println(")")
			default:
				fmt.Printf("  v%d = %s", in.Result.ID, in.Op)
				for i, a := range in.Args {
					if i == 0 {
						fmt.Print(" ")
					} else {
						fmt.Print(", ")
					}
					fmt.Printf("v%d", a.ID)
				}
				fmt.Println()
			}
		}
		fmt.Printf("  return v%d\n}\n", fn.Result.ID)
	}
}
