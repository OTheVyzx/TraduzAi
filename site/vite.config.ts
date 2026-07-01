import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(siteRoot, "..");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^react$/, replacement: path.resolve(siteRoot, "node_modules/react") },
      { find: /^react-dom$/, replacement: path.resolve(siteRoot, "node_modules/react-dom") },
      { find: /^react-router$/, replacement: path.resolve(siteRoot, "node_modules/react-router") },
      { find: /^react-router-dom$/, replacement: path.resolve(siteRoot, "node_modules/react-router-dom") },
      { find: /^zustand$/, replacement: path.resolve(siteRoot, "node_modules/zustand") },
      { find: /^konva$/, replacement: path.resolve(siteRoot, "node_modules/konva") },
      { find: /^react-konva$/, replacement: path.resolve(siteRoot, "node_modules/react-konva") },
    ],
    dedupe: ["react", "react-dom", "react-router", "react-router-dom", "zustand", "konva", "react-konva"],
  },
  server: {
    fs: {
      allow: [siteRoot, projectRoot],
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8787",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
