import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const host = process.env.TAURI_DEV_HOST;
const workspaceRoot = process.cwd();
const realWorkspaceRoot = fs.realpathSync(workspaceRoot);
const allowedRoots = Array.from(new Set([workspaceRoot, realWorkspaceRoot]));
const ignoredGlobs = [
  "**/src-tauri/**",
  "**/vision-worker/**",
  "**/pipeline/**",
  "**/debug_runs/**",
  "**/testes/**",
  "**/.venv/**",
  "**/.toolvenv/**",
];

export default defineConfig(async () => ({
  // Keep Vite anchored to the real filesystem path so Windows junctions
  // like T:\traduzai -> D:\traduzai don't produce absolute HTML emit names.
  root: realWorkspaceRoot,
  envDir: realWorkspaceRoot,
  publicDir: path.join(realWorkspaceRoot, "public"),
  cacheDir: path.join(realWorkspaceRoot, "node_modules", ".vite"),
  plugins: [react()],
  clearScreen: false,
  optimizeDeps: {
    // Restrict the initial crawl to the actual frontend entry so generated
    // HTML/MJS files inside Rust targets do not get treated as app inputs.
    entries: [path.join(realWorkspaceRoot, "index.html")],
  },
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    watch: { ignored: ignoredGlobs },
    fs: {
      allow: allowedRoots,
      deny: ["**/vision-worker/**", "**/src-tauri/target/**"],
    },
  },
}));
