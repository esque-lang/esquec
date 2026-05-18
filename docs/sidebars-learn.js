// Learn track — tutorial-first. This is the front door for newcomers.
/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  learn: [
    'index',
    {
      type: 'category',
      label: 'Getting started',
      collapsed: false,
      items: [
        'install',
        'hello-world',
        'tour',
      ],
    },
    {
      type: 'category',
      label: 'Tutorial',
      collapsed: false,
      items: [
        'tutorial/values-and-types',
        'tutorial/functions',
        'tutorial/tensors',
        'tutorial/pipelines-and-reductions',
        'tutorial/loop-primitives',
        'tutorial/pattern-matching',
        'tutorial/control-flow',
        'tutorial/printing-and-io',
      ],
    },
    {
      type: 'category',
      label: 'Worked examples',
      collapsed: true,
      items: [
        'examples/index',
        'examples/exit-status',
        'examples/recursion',
        'examples/dot-product',
        'examples/euclidean-distance',
        'examples/scan-prefix-sum',
        'examples/iterate-until',
      ],
    },
    {
      type: 'category',
      label: 'Going further',
      collapsed: true,
      items: [
        'going-further/effects-and-io',
        'going-further/performance',
        'going-further/cli',
        'going-further/faq',
      ],
    },
  ],
};
export default sidebars;
