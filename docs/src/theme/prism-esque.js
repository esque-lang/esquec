// Prism language definition for esque
// (https://esque-lang.github.io/esquec/).
//
// Mirrors the token classes in tree-sitter-esque/queries/highlights.scm so
// the rendered docs match what contributors see in their editor. Regex-based,
// so context-sensitive cases (e.g. distinguishing a user-named `dot` call
// from a local binding) are intentionally collapsed — the tree-sitter
// grammar and esque-lsp remain the source of truth for editor tooling.

(function (Prism) {
  Prism.languages.esque = {
    'comment': [
      // Block comments are nestable in esque, but Prism's regex engine
      // cannot match nested constructs; we approximate with a
      // non-greedy match, exactly as tree-sitter-esque does. The
      // compiler and LSP enforce real nesting.
      {
        pattern: /\/\*[\s\S]*?\*\//,
        greedy: true,
      },
      {
        pattern: /\/\/.*/,
        greedy: true,
      },
    ],
    'string': {
      pattern: /"(?:\\.|[^"\\\n])*"/,
      greedy: true,
    },
    'char': {
      pattern: /'(?:\\.|[^'\\\n])'/,
      greedy: true,
      alias: 'string',
    },
    'attribute': {
      pattern: /@[A-Za-z_]\w*/,
      alias: 'symbol',
    },
    'keyword': /\b(?:fn|let|mut|return|if|else|match|as|in)\b/,
    'boolean': /\b(?:true|false)\b/,
    'builtin':
      /\b(?:i8|i16|i32|i64|u8|u16|u32|u64|f32|f64|bool|unit|nat)\b/,
    'function': {
      pattern:
        /\b(?:tabulate|scan|iterate_until|iterate|each|print_(?:i32|f32|str))\b/,
    },
    'number':
      /\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?(?:_[iuf](?:8|16|32|64))?\b/,
    'operator':
      /\.\.=?|\|>|->|=>|==|!=|<=|>=|&&|\|\||\.[+\-*/%]|[+*]\/|[+\-*/%=<>!@|]/,
    'punctuation': /[{}[\]();,:]/,
  };
}(Prism));
