/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Block colors — must stay in sync with src/data/architectureData.js
        embed: '#3b82f6', // blue — embedding / input processing
        selfattn: '#c084fc', // light purple — self-attention
        maskattn: '#e879f9', // magenta — masked self-attention (causal)
        crossattn: '#4c1d95', // dark purple/indigo — cross-attention
        addnorm: '#22c55e', // green — Add & Norm
        ffn: '#f97316', // orange — feed-forward network
        output: '#eab308' // gold — output / softmax
      }
    }
  },
  plugins: []
}
