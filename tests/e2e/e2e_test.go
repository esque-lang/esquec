package e2e

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// compiler is the path to the esquec binary built once by TestMain
// and shared by every TestE2E subtest.
var compiler string

func TestMain(m *testing.M) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		os.Exit(m.Run())
	}
	tmp, err := os.MkdirTemp("", "esquec-e2e-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "TestMain: mktemp: %v\n", err)
		os.Exit(2)
	}
	compiler = filepath.Join(tmp, "esquec")
	if out, err := exec.Command("go", "build", "-o", compiler, "../../cmd/esquec").CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "TestMain: go build esquec: %v\n%s", err, out)
		os.RemoveAll(tmp)
		os.Exit(2)
	}
	code := m.Run()
	os.RemoveAll(tmp)
	os.Exit(code)
}

type expectations struct {
	exit         int
	hasExit      bool
	stdout       string
	hasStdout    bool
	compileError string
	hasCompile   bool
	skip         string
}

var (
	directiveRE = regexp.MustCompile(`^\s*(?://|#)\s*expect\s+([a-z][a-z-]*)\s+(.*?)\s*$`)
	commentRE   = regexp.MustCompile(`^\s*(?://|#)`)
)

func parseDirectives(t *testing.T, src string) expectations {
	t.Helper()
	f, err := os.Open(src)
	if err != nil {
		t.Fatalf("open %s: %v", src, err)
	}
	defer f.Close()

	var exp expectations
	any := false
	sc := bufio.NewScanner(f)
	for line := 0; sc.Scan(); {
		line++
		raw := sc.Text()
		if strings.TrimSpace(raw) == "" {
			continue
		}
		if !commentRE.MatchString(raw) {
			break // first non-comment, non-blank line ends the header
		}
		m := directiveRE.FindStringSubmatch(raw)
		if m == nil {
			continue // plain comment in header is not a directive
		}
		any = true
		switch name, tail := m[1], m[2]; name {
		case "exit":
			if exp.hasExit {
				t.Fatalf("%s:%d: duplicate `expect exit`", src, line)
			}
			n, err := strconv.Atoi(tail)
			if err != nil || n < 0 {
				t.Fatalf("%s:%d: `expect exit` wants non-negative int, got %q", src, line, tail)
			}
			exp.exit, exp.hasExit = n, true
		case "stdout":
			if exp.hasStdout {
				t.Fatalf("%s:%d: duplicate `expect stdout`", src, line)
			}
			s, err := strconv.Unquote(tail)
			if err != nil {
				t.Fatalf("%s:%d: `expect stdout` wants Go-quoted string, got %q: %v", src, line, tail, err)
			}
			exp.stdout, exp.hasStdout = s, true
		case "compile-error":
			if exp.hasCompile {
				t.Fatalf("%s:%d: duplicate `expect compile-error`", src, line)
			}
			s, err := strconv.Unquote(tail)
			if err != nil {
				t.Fatalf("%s:%d: `expect compile-error` wants Go-quoted string, got %q: %v", src, line, tail, err)
			}
			exp.compileError, exp.hasCompile = s, true
		case "skip":
			if tail == "" {
				t.Fatalf("%s:%d: `expect skip` requires a reason", src, line)
			}
			exp.skip = tail
		default:
			t.Fatalf("%s:%d: unknown directive `expect %s`", src, line, name)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", src, err)
	}
	if !any {
		t.Fatalf("%s: no `expect` directives found; every fixture must declare at least one", src)
	}
	if exp.hasCompile && (exp.hasExit || exp.hasStdout) {
		t.Fatalf("%s: `expect compile-error` is mutually exclusive with `expect exit` and `expect stdout`", src)
	}
	if exp.skip == "" && !exp.hasCompile && !exp.hasExit {
		t.Fatalf("%s: fixture must declare `expect exit N`, `expect compile-error ...`, or `expect skip ...`", src)
	}
	return exp
}

func TestE2E(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skipf("e2e tests require linux/amd64; got %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	fixtures, err := filepath.Glob("*.esq")
	if err != nil {
		t.Fatal(err)
	}
	if len(fixtures) == 0 {
		t.Fatal("no .esq fixtures discovered in tests/e2e")
	}
	for _, src := range fixtures {
		src := src
		name := strings.TrimSuffix(filepath.Base(src), ".esq")
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			exp := parseDirectives(t, src)
			if exp.skip != "" {
				t.Skip(exp.skip)
			}
			runFixture(t, src, name, exp)
		})
	}
}

func runFixture(t *testing.T, src, name string, exp expectations) {
	t.Helper()
	abs, err := filepath.Abs(src)
	if err != nil {
		t.Fatalf("abs %s: %v", src, err)
	}
	if exp.hasCompile {
		out, err := exec.Command(compiler, "check", abs).CombinedOutput()
		if err == nil {
			t.Fatalf("compile: expected error containing %q, got success:\n%s", exp.compileError, out)
		}
		if _, ok := err.(*exec.ExitError); !ok {
			t.Fatalf("compile: expected ExitError, got %T: %v\n%s", err, err, out)
		}
		if !strings.Contains(string(out), exp.compileError) {
			t.Fatalf("compile: stderr does not contain %q:\n%s", exp.compileError, out)
		}
		return
	}
	exe := filepath.Join(t.TempDir(), name)
	if out, err := exec.Command(compiler, "build", "-o", exe, abs).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	stdout, runErr := exec.Command(exe).Output()
	gotExit := 0
	if runErr != nil {
		exitErr, ok := runErr.(*exec.ExitError)
		if !ok {
			t.Fatalf("run: %v (stdout=%q)", runErr, stdout)
		}
		gotExit = exitErr.ExitCode()
	}
	if gotExit != exp.exit {
		t.Fatalf("run: exit code = %d, want %d (stdout=%q)", gotExit, exp.exit, stdout)
	}
	if exp.hasStdout {
		if got := string(stdout); got != exp.stdout {
			t.Errorf("run: stdout = %q, want %q", got, exp.stdout)
		}
	}
}
