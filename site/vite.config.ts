import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const prerenderModule = require("vite-plugin-prerender");
const prerender = prerenderModule.default ?? prerenderModule;

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
