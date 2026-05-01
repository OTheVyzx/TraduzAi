/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B0B12",
        panel: "#151522",
        primary: "#7C5CFF",
        cyan: "#22D3EE",
        soft: "#9CA3AF",
      },
    },
  },
  plugins: [],
};
