// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'esque',
  tagline: 'A statically typed, tensor-primitive systems language',
  favicon: 'img/favicon.svg',

  future: {
    v4: true,
  },

  url: process.env.DOCUSAURUS_URL ?? 'https://esque-lang.github.io',
  baseUrl: process.env.DOCUSAURUS_BASE_URL ?? '/',

  organizationName: 'esque-lang',
  projectName: 'esquec',

  onBrokenLinks: 'warn',

  // Treat .md as plain CommonMark; only .mdx files are parsed as MDX. This
  // lets us use raw `<` characters (e.g. `T[N <= 32]`, `f32 < f64`) in
  // tutorial and reference prose without escaping them as JSX.
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Primary "Learn" track is the default classic-preset docs instance,
  // mounted at the site root for prominence. The "Reference" track is a
  // second @docusaurus/plugin-content-docs instance mounted at /reference;
  // it is intentionally de-emphasized in the navbar so newcomers land on
  // tutorials by default.
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'learn',
          routeBasePath: '/',
          sidebarPath: './sidebars-learn.js',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'reference',
        path: 'reference',
        routeBasePath: 'reference',
        sidebarPath: './sidebars-reference.js',
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.png',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'esque',
        logo: {
          alt: 'esque',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'index',
            position: 'left',
            label: 'Learn',
          },
          {
            type: 'doc',
            docId: 'index',
            docsPluginId: 'reference',
            position: 'right',
            label: 'Reference',
          },
          {
            href: 'https://github.com/esque-lang/esquec',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Learn',
            items: [
              {label: 'Hello, world', to: '/hello-world'},
              {label: 'Tour of esque', to: '/tour'},
              {label: 'Worked examples', to: '/examples'},
            ],
          },
          {
            title: 'Reference',
            items: [
              {label: 'Reference index', to: '/reference/'},
              {label: 'Spec', to: '/reference/spec/overview'},
              {label: 'Planned features', to: '/reference/planned/overview'},
            ],
          },
          {
            title: 'Project',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/esque-lang/esquec',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} The esque project. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'go', 'rust', 'json'],
      },
    }),
};

export default config;
