/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        night: "#0b0d12",
        panel: "#12151d",
        edge: "#1f2430",
      },
    },
  },
  plugins: [],
};
