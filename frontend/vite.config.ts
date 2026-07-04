import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Build output is committed into the python package so
// `python -m cyberwheel frontend <port>` serves the UI with no node install.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../cyberwheel/server/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://localhost:8123",
    },
  },
});
