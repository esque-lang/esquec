// Reference track — fully fleshed out language reference, compiler internals,
// the formal-ish spec, and a planned-features subsection. Intentionally
// de-emphasized in the navbar so newcomers do not land here first.
/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  reference: [
    'index',
    {
      type: 'category',
      label: 'Language reference',
      collapsed: false,
      items: [
        'language/lexical',
        'language/operators',
        'language/types',
        'language/functions',
        'language/tensors',
        'language/loop-primitives',
        'language/pattern-matching',
        'language/intrinsics',
        'language/grammar',
      ],
    },
    {
      type: 'category',
      label: 'Compiler internals',
      collapsed: true,
      items: [
        'internals/pipeline',
        'internals/lexer',
        'internals/parser',
        'internals/types',
        'internals/ceir',
        'internals/mir',
        'internals/x86',
        'internals/elf-and-linking',
        'internals/diagnostics',
      ],
    },
    {
      type: 'category',
      label: 'Specification',
      collapsed: true,
      items: [
        'spec/overview',
        'spec/lexical',
        'spec/types',
        'spec/expressions',
        'spec/functions',
        'spec/tensors',
        'spec/modules',
      ],
    },
    {
      type: 'category',
      label: 'Planned features',
      collapsed: true,
      items: [
        'planned/overview',
        'planned/effects',
        'planned/strings',
        'planned/extended-numerics',
        'planned/traits',
        'planned/linear-types',
        'planned/kernel-dsl',
        'planned/gpu-backend',
        'planned/autodiff',
        'planned/roadmap',
      ],
    },
  ],
};
export default sidebars;
