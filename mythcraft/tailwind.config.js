/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0D0F14',
        surface: '#161A22',
        text: '#EDEDE6',
        textDim: '#9096A3',
        accent: '#E8B84B',
        busted: '#D1495B',
        confirmed: '#3FA796',
        mixed: '#E0954D',
        hairline: 'rgba(237,237,230,0.09)',
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        body: ['"Work Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      maxWidth: {
        // The content well every page reads inside.
        well: '40rem',
      },
      keyframes: {
        // The one bold moment: the verdict lands like a rubber stamp.
        stamp: {
          '0%': { opacity: '0', transform: 'scale(1.7) rotate(6deg)' },
          '55%': { opacity: '1', transform: 'scale(0.94) rotate(-3deg)' },
          '100%': { opacity: '1', transform: 'scale(1) rotate(-1.5deg)' },
        },
      },
      animation: {
        stamp: 'stamp 620ms cubic-bezier(0.2, 0.9, 0.25, 1) both',
      },
    },
  },
  plugins: [],
};
