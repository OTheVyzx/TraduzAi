import baseConfig from "../tailwind.config.js";

export default {
  ...baseConfig,
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "../src/pages/**/*.{js,ts,jsx,tsx}",
    "../src/editor-shared/**/*.{js,ts,jsx,tsx}",
    "../src/components/editor/**/*.{js,ts,jsx,tsx}",
    "../src/lib/**/*.{js,ts,jsx,tsx}",
  ],
};
