import { spawn } from "node:child_process";
import { realpathSync } from "node:fs";
import path from "node:path";

const realWorkspaceRoot = realpathSync(process.cwd());
const viteBin = path.join(realWorkspaceRoot, "node_modules", "vite", "bin", "vite.js");
const args = process.argv.slice(2);

const child = spawn(process.execPath, [viteBin, ...args], {
  cwd: realWorkspaceRoot,
  stdio: "inherit",
  env: process.env,
});

child.on("error", (error) => {
  console.error("[run-vite] falha ao iniciar Vite:", error);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
