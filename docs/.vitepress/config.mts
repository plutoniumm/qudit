import { defineConfig } from 'vitepress';

export default defineConfig({
  title: "qudit",
  description: "documentation for qudit quantum simulator",
  markdown: {
    math: true
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
          { text: 'Circuit', link: '/circuit' }
        ]
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
