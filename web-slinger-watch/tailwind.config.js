/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: '#0a1628',
        grid: '#1e3a5f',
        'marker-pink': '#ff6b8a',
        'marker-green': '#4ade80',
        alert: '#f5b878',
        teal: '#2dd4bf',
        'text-primary': '#e8f4f8',
        'text-muted': '#7d97ad',
      },
      fontFamily: {
        display: ['"Press Start 2P"', 'cursive'],
        body: ['"Work Sans"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
