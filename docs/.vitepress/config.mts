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
          { text: 'Noisy Circuits', link: '/noisy' },
          { text: 'Circuit Cutting', link: '/cutting' },
          { text: 'QUBO', link: '/qubo' },
          { text: 'Error Correction', link: '/qec' },
          { text: 'Noise Channels', link: '/noise' },
          { text: 'Metrics', link: '/metrics' },
          { text: 'Algorithms', link: '/algo' },
        ]
      },
      {
        text: 'Performance',
        items: [
          { text: 'Overview', link: '/benchmarks' },
          { text: 'Ideal Circuit', link: '/bench-circuit' },
          { text: 'Noisy Circuits', link: '/bench-noisy' },
          { text: 'QEC Recovery', link: '/bench-qec' },
          { text: 'Gradient Descent', link: '/bench-gd' },
          { text: 'QUBO / QAOA', link: '/bench-qubo' },
          { text: 'Metrics', link: '/bench-metrics' },
        ],
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

    search: {
      provider: 'local'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/plutoniumm/qudit' },
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
