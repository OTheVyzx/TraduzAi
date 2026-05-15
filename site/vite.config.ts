import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import prerender from "vite-plugin-prerender";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(siteRoot, "..");

export default defineConfig({
  plugins: [
    react(),
    prerender({
      staticDir: path.join(path.dirname(fileURLToPath(import.meta.url)), "dist"),
      routes: ["/", "/legal"],
    }),
  ],
  resolve: {
    dedupe: ["react", "react-dom", "zustand", "konva", "react-konva"],
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
