/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        dusk: {
          900: '#14172E',
          800: '#1B1F3B',
          700: '#23294E',
        },
        marigold: '#FFA630',
        parrot: '#3C9D5C',
        kite: '#E84855',
        offwhite: '#F5F0E6',
      },
      fontFamily: {
        display: ['"Baloo 2"', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
