import { spawn } from "node:child_process";
import { realpathSync } from "node:fs";
import path from "node:path";
import { cleanupWorkspaceDevProcesses } from "./tauri-dev-preflight.mjs";

const realWorkspaceRoot = realpathSync(process.cwd());
const tauriBin = path.join(realWorkspaceRoot, "node_modules", "@tauri-apps", "cli", "tauri.js");
const args = process.argv.slice(2);

if (process.platform === "win32" && args.includes("dev")) {
  const cleaned = await cleanupWorkspaceDevProcesses(realWorkspaceRoot);
  if (cleaned.length > 0) {
    console.log(`[run-tauri] Limpando processos de dev antigos do workspace: ${cleaned.join(", ")}`);
  }
}

const child = spawn(process.execPath, [tauriBin, ...args], {
  cwd: realWorkspaceRoot,
  stdio: "inherit",
  env: process.env,
});

child.on("error", (error) => {
  console.error("[run-tauri] falha ao iniciar Tauri CLI:", error);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
