/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#07090f",
          900: "#0b0e16",
          850: "#0f131d",
          800: "#141927",
          700: "#1c2333",
          600: "#273044",
          500: "#39445c",
        },
        accent: {
          DEFAULT: "#5eb0ff",
          dim: "#3b82c4",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "JetBrains Mono",
          "Menlo",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
