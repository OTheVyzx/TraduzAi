import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

function vitePath(path: string) {
  return path.replace(/\\/g, "/");
}

const currentEditorBackendShim = vitePath(resolve(__dirname, "src", "shims", "currentEditorBackend.ts"));
const currentEditorStorePath = vitePath(resolve(__dirname, "..", "src", "lib", "stores", "editorStore.ts"));
const currentEditorBackendPath = vitePath(resolve(__dirname, "..", "src", "lib", "editorBackend.ts"));

export default defineConfig({
  plugins: [
    {
      name: "studio-current-editor-backend-shim",
      enforce: "pre",
      resolveId(source, importer) {
        const normalizedSource = vitePath(source);
        const normalizedImporter = importer ? vitePath(importer) : "";
        if (normalizedSource === currentEditorBackendPath || normalizedSource.endsWith("/src/lib/editorBackend")) {
          return currentEditorBackendShim;
        }
        if (normalizedImporter === currentEditorStorePath && normalizedSource === "../editorBackend") {
          return currentEditorBackendShim;
        }
        return null;
      },
    },
    react(),
  ],
  publicDir: resolve(__dirname, "..", "public"),
  server: {
    host: "127.0.0.1",
    port: 1430,
    fs: {
      allow: [resolve(__dirname), resolve(__dirname, "..")],
    },
  },
});
