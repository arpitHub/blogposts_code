/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: '#0a1628',
        // Borough gradient stops: deep center -> slightly lifted edge.
        'navy-deep': '#081221',
        'navy-lift': '#16293f',
        'navy-hot': '#1d3a52',
        grid: '#1e3a5f',
        'grid-fine': '#173049',
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
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '0.25', transform: 'scale(1)' },
          '50%': { opacity: '0.6', transform: 'scale(1.18)' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        shake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '20%': { transform: 'translateX(-5px)' },
          '40%': { transform: 'translateX(5px)' },
          '60%': { transform: 'translateX(-3px)' },
          '80%': { transform: 'translateX(3px)' },
        },
        'success-pop': {
          '0%': { opacity: '0', transform: 'scale(0.7)' },
          '40%': { opacity: '1', transform: 'scale(1.08)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
      },
      animation: {
        'pulse-glow': 'pulse-glow 2.6s ease-in-out infinite',
        scanline: 'scanline 7s linear infinite',
        shake: 'shake 380ms ease-in-out',
        'success-pop': 'success-pop 320ms cubic-bezier(0.2, 0.8, 0.3, 1) both',
      },
    },
  },
  plugins: [],
};
