/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#0F0F14",
          secondary: "#1A1A24",
          tertiary: "#242434",
          hover: "#2A2A3E",
        },
        accent: {
          purple: "#7C5CFF",
          "purple-light": "#9B82FF",
          "purple-dark": "#5A3ED4",
          cyan: "#00D4FF",
          pink: "#FF5CAA",
        },
        text: {
          primary: "#E8E8F0",
          secondary: "#9898B0",
          muted: "#686882",
        },
        status: {
          success: "#4ADE80",
          warning: "#FBBF24",
          error: "#F87171",
          info: "#60A5FA",
        },
      },
      fontFamily: {
        sans: ['"Geist Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
      borderRadius: {
        DEFAULT: "8px",
        lg: "12px",
        xl: "16px",
      },
    },
  },
  plugins: [],
};
