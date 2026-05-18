<h1 align="center">
  <img src="docs/static/img/logo.svg" alt="esque" width="120" /><br/>
  esquec
</h1>

<p align="center">
  A statically typed, tensor-primitive systems language. The compiler is a
  single Go binary that emits ELF x86-64 Linux executables directly —
  no LLVM, no runtime to link.
</p>

<p align="center">
  <a href="https://github.com/esque-lang/esquec/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/esque-lang/esquec/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/esque-lang/esquec/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/esque-lang/esquec/actions/workflows/docs.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/esque-lang/esquec/actions/workflows/release.yml"><img alt="Release" src="https://github.com/esque-lang/esquec/actions/workflows/release.yml/badge.svg"></a>
  <a href="https://github.com/esque-lang/esquec/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/esque-lang/esquec?sort=semver"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/esque-lang/esquec"></a>
  <a href="go.mod"><img alt="Go version" src="https://img.shields.io/github/go-mod/go-version/esque-lang/esquec"></a>
</p>

<p align="center">
  <a href="https://esque-lang.github.io/esquec/"><strong>Welcome</strong></a>
  &nbsp;·&nbsp;
  <a href="https://esque-lang.github.io/esquec/tour">Tour</a>
  &nbsp;·&nbsp;
  <a href="https://esque-lang.github.io/esquec/reference/">Reference</a>
</p>

## Highlights

- **Tensors are values.** A function takes `f32[3, N]` the same way it
  takes `i32`; shapes are part of the type.
- **Loop primitives over `for`/`while`.** `tabulate`, `scan`,
  `iterate_until`, and operators like `+/` and `.*` name the patterns
  numeric code actually uses.
- **Single static toolchain.** Parse → typecheck → CEIR → MIR → x86-64
  → ELF, all in one Go binary. The only external program invoked at
  compile time is `ld`.

## Install

Requires Go 1.25+ and a Linux x86-64 host with a working `ld`.

```bash
git clone https://github.com/esque-lang/esquec.git
cd esquec
go build -o esquec ./cmd/esquec
```

Or grab a prebuilt binary from the
[latest release](https://github.com/esque-lang/esquec/releases/latest).

Nix users:

```bash
nix build github:esque-lang/esquec
```

## Quickstart

```esque
fn dot[N](x: f32[N], y: f32[N]) -> f32 = +/(x .* y)

fn main() -> i32 = {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = dot(a, b);
    result as i32
}
```

```bash
./esquec build hello.esq -o hello
./hello; echo $?
# 32
```

## Tests

```bash
go test ./...
```

Unit tests live alongside each package under `internal/`; end-to-end
tests compile real `.esq` programs and assert on exit codes and stdout
under `tests/e2e/`.

## Documentation

The full site — tutorial, reference, and spec — is published to
**[esque-lang.github.io/esquec](https://esque-lang.github.io/esquec/)**.
Sources live under [`docs/`](docs/) and redeploy on every push to
`main`.

## Releases

Push a tag matching `v*` (e.g. `git tag v0.1.0 && git push --tags`)
and the release workflow cross-builds `linux/amd64` and `linux/arm64`
tarballs, uploads SHA-256 sums, and publishes a GitHub Release with
auto-generated notes.

## License

[MIT](LICENSE) © Donnis Moore.
