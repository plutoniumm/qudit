import Code from "../code/files.json" assert { type: "json" };
import { defineConfig } from 'vitepress';

const isDev = process.env.NODE_ENV === 'development';

export default defineConfig({
  base: isDev ? '/' : '/qudit/',
  title: "qudit",
  description: "documentation for qudit quantum simulator",
  head: [
    ['link', { rel: 'icon', href: '/icons/favi.svg', type: 'image/svg+xml' }],
    ['link', { rel: 'apple-touch-icon', href: '/icons/favi.svg', type: 'image/svg+xml' }],
  ],
  markdown: {
    math: true,
    lineNumbers: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
    ],

    sidebar: [
      {
        text: 'Usage',
        items: [
          { text: 'Quickstart', link: '/quickstart' },
          { text: 'Primitives', link: '/primitives' },
          { text: 'Circuit', link: '/circuit' },
          { text: 'QUBO', link: '/qubo' },
          { text: 'Error Correction', link: '/qec' },
          { text: 'Noise Channels', link: '/noise' },
          { text: 'Metrics', link: '/metrics' },
          { text: 'Algorithms', link: '/algo' },
        ]
      },
      {
        text: 'Testing',
        items: [
          { text: 'Gate tests', link: '/tests/gates' },
        ],
      },
      {
        text: 'Code',
        items: Code
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/plutoniumm/qudit' },
      { icon: 'ieee', link: 'https://ieeexplore.ieee.org/abstract/document/11333840' }
    ]
  },
  vite: {
    server: {
      port: 3000,
      fs: {
        strict: false
      },
    }
  }
})
